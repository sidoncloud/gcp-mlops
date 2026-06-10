# =============================================================================
# Setup and the Hallucination Problem
# =============================================================================
"""
Reducing hallucination with Gemini on Vertex AI.

A large language model answers from a frozen snapshot of training data. It has
no knowledge of events after its cutoff, and it has never seen your company's
private documents. When asked about either, it will often produce a confident
but wrong answer. This is hallucination.

There are two complementary fixes, and this lab covers both:

  1. Grounding with Google Search
     Let the model issue live web searches and answer from the results. Best
     for recent, public, fast-changing facts (news, prices, scores, releases).
     Every claim comes back with citations you can verify.

  2. Retrieval Augmented Generation (RAG) with the Vertex AI RAG Engine
     Index your OWN documents into a corpus, then let the model retrieve the
     relevant passages at query time and answer from them. Best for private
     knowledge the web does not have (internal wikis, contracts, manuals).

Grounding reaches outward to the public web. RAG reaches inward to your private
corpus. Production systems frequently use both.
"""

from google import genai
from google.genai import types

PROJECT_ID = "your-project-id"  # replace with your Google Cloud project ID
LOCATION = "us-central1"
MODEL = "gemini-2.5-flash"

# vertexai=True routes the SDK to Vertex AI instead of the public Gemini API,
# so it uses your project's IAM, quota, and data-governance boundary.
client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)


# =============================================================================
# Baseline: Asking Without Grounding
# =============================================================================

# A question whose true answer changes over time and lives past the model's
# training cutoff. With no tools, the model can only guess from stale memory.
QUESTION = "Who won the most recent Formula 1 World Drivers' Championship, and with how many points?"

baseline = client.models.generate_content(
    model=MODEL,
    contents=QUESTION,
)

print("=== BASELINE (no grounding) ===")
print(baseline.text)
# WARNING: this answer reflects the training cutoff only. For a recent event it
# may be outdated or entirely hallucinated, and it carries no sources to check.


# =============================================================================
# Grounding with Google Search
# =============================================================================

# Attaching the GoogleSearch tool lets Gemini decide to run web searches and
# answer from what it finds. The model also returns grounding metadata so you
# can audit exactly which sources and search queries backed the answer.
grounded = client.models.generate_content(
    model=MODEL,
    contents=QUESTION,
    config=types.GenerateContentConfig(
        tools=[types.Tool(google_search=types.GoogleSearch())],
    ),
)

print("\n=== GROUNDED (Google Search) ===")
print(grounded.text)

# Grounding metadata lives on the candidate, not on the response root.
metadata = grounded.candidates[0].grounding_metadata

if metadata is not None:
    # The actual web searches the model ran to answer the question.
    if metadata.web_search_queries:
        print("\nSearch queries used:")
        for query in metadata.web_search_queries:
            print(f"  - {query}")

    # The web pages cited as evidence. Each chunk's .web holds title and uri.
    if metadata.grounding_chunks:
        print("\nCited sources:")
        for chunk in metadata.grounding_chunks:
            if chunk.web is not None:
                print(f"  - {chunk.web.title}: {chunk.web.uri}")
else:
    # If metadata is None the model chose not to search (it judged the
    # question answerable without the web). That is expected behavior.
    print("\n(Model answered without invoking search.)")


# =============================================================================
# When the Answer Lives in Your Private Docs (RAG)
# =============================================================================

# Google Search grounding cannot help when the truth is not public: your
# onboarding handbook, a signed contract, last quarter's internal postmortem.
#
# RAG fixes this. The Vertex AI RAG Engine lets you:
#   1. Create a corpus (a managed vector index).
#   2. Import your documents into it (it chunks and embeds them for you).
#   3. Attach the corpus to Gemini as a retrieval tool at query time.
#
# At query time the model retrieves the most relevant passages from YOUR corpus
# and answers from them, so responses stay grounded in private knowledge.


# =============================================================================
# Create a RAG Corpus and Import Documents from GCS
# =============================================================================

# Corpus management (create / import) lives in the vertexai.rag preview module,
# which ships separately from google-genai. This is the rag module, NOT the
# retired vertexai.generative_models module, so it is safe to use here.
#
# WHY you need this pip install: a stock Workbench/Colab environment usually
# has google-cloud-aiplatform installed without the rag extra, so importing
# vertexai.rag will fail with ModuleNotFoundError until you add it. Run this
# once at the top of the notebook, then restart the kernel so the freshly
# installed rag module is picked up:
#
#   !pip install --upgrade "google-cloud-aiplatform[rag]"
#
# The rag module is what gives you create_corpus, import_files, and the
# managed embedding configuration we use below.
import vertexai
from vertexai import rag

# RAG_LOCATION is the region the corpus lives in. It does NOT have to match
# your GCS bucket region. We deliberately point it at us-west1 because the
# Vertex AI RAG Engine has capacity restrictions on us-central1, us-east1,
# and us-east4 for new GCP projects (Spanner-backed mode is allowlist-only
# in those three regions), and trying to create a corpus there fails with
# INVALID_ARGUMENT. us-west1 has no such restriction.
RAG_LOCATION = "us-west1"
vertexai.init(project=PROJECT_ID, location=RAG_LOCATION)

# Create the managed corpus. The embedding model turns each chunk into a
# vector so the engine can retrieve by meaning, not just keyword match.
# We use the default backend, which works cleanly in us-west1 with no
# additional vector_db config required.
corpus = rag.create_corpus(
    display_name="company-knowledge-base",
    backend_config=rag.RagVectorDbConfig(
        rag_embedding_model_config=rag.RagEmbeddingModelConfig(
            vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
                publisher_model="publishers/google/models/text-embedding-005"
            )
        ),
    ),
)
print("\nCreated corpus:", corpus.name)

# Import your documents. The lab ships a sample file under sample-docs/ named
# remote-work-policy.txt. Upload it to a GCS bucket you own with:
#   gsutil cp sample-docs/remote-work-policy.txt \
#     gs://YOUR_BUCKET/company-docs/remote-work-policy.txt
# Then replace the path below with your own bucket/prefix. The RAG Engine
# accepts PDF, TXT, DOCX, and a few other formats.
rag.import_files(
    corpus.name,
    paths=["gs://your-bucket-name/company-docs/"],  # replace with your own GCS path
    transformation_config=rag.TransformationConfig(
        chunking_config=rag.ChunkingConfig(chunk_size=512, chunk_overlap=100),
    ),
)
print("Import started. Files are chunked and embedded asynchronously.")


# =============================================================================
# Query the Corpus with Gemini
# =============================================================================

# We query through google-genai by attaching the corpus as a retrieval tool.
# types.VertexRagStore points the retrieval at our corpus; Gemini fetches the
# top matching chunks and grounds its answer on them.
PRIVATE_QUESTION = "What is our company's policy on remote work eligibility?"

rag_tool = types.Tool(
    retrieval=types.Retrieval(
        vertex_rag_store=types.VertexRagStore(
            rag_resources=[
                types.VertexRagStoreRagResource(
                    rag_corpus=corpus.name,  # full resource name from Cell 5
                )
            ],
            rag_retrieval_config=types.RagRetrievalConfig(
                top_k=5,  # how many chunks to pull into the prompt
            ),
        )
    )
)

rag_answer = client.models.generate_content(
    model=MODEL,
    contents=PRIVATE_QUESTION,
    config=types.GenerateContentConfig(tools=[rag_tool]),
)

print("\n=== RAG (private corpus) ===")
print(rag_answer.text)

# The retrieved passages that grounded the answer show up here, letting you
# trace the response back to the exact source document.
rag_metadata = rag_answer.candidates[0].grounding_metadata
if rag_metadata is not None and rag_metadata.grounding_chunks:
    print("\nRetrieved from corpus:")
    for chunk in rag_metadata.grounding_chunks:
        if chunk.retrieved_context is not None:
            print(f"  - {chunk.retrieved_context.title}: {chunk.retrieved_context.uri}")


# =============================================================================
# Grounding vs RAG in Production
# =============================================================================

# Grounding with Google Search wins on web freshness: recent, public, broad
# facts, with no infrastructure to maintain. But it cannot see private data and
# you do not control the source set.
#
# RAG wins on private knowledge: it answers from documents you own and curate,
# with full control over what the model can cite. The tradeoff is that you must
# build and keep the corpus current.
#
# In production they are complementary. A support assistant might ground product
# pricing on live search while answering account-specific questions from a RAG
# corpus. Either way, both approaches return citations, which is what turns a
# plausible-sounding guess into a verifiable, trustworthy answer.
