# NL2SQL - Natural Language to SQL System

A lightweight NL2SQL (Natural Language to SQL) prototype that enables users to query their databases using plain English. Ask questions, uncover KPIs, and extract actionable insights—no SQL expertise required. Built with semantic schema search and understanding, safe query execution, and full transparency into how queries are generated.

---


## Table of Contents

- [Setup](#setup)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Overview](#overview)
  - [Key Features](#key-features)
- [References](#references)
- [Example Queries](#example-queries)
- [Architecture](#architecture)
  - [Layer Responsibilities](#layer-responsibilities)
- [Tech Stack](#tech-stack)
- [Pipeline](#pipeline)
  - [Schema Embedding Strategy](#schema-embedding-strategy)
  - [Design Decisions](#design-decisions)
- [Security](#security)
- [License](#license)

---


## Setup

### Prerequisites

- Python 3.11+
- Poetry (dependency management)
- PostgreSQL 15+ with pgvector extension
- OpenRouter/Anthropic/OpenAI API key [for LLM and Embedding Models]
- Supabase Project [Free Tier]

### 1. Clone and Install

```bash
# Clone repository
git clone <repository-url>
cd nl2sql

# Install Poetry (if not installed)
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Activate virtual environment
poetry shell


```

### 2. Database Setup

#### Option A: Supabase (Recommended)

1. Create a Supabase project at https://supabase.com
2. Enable pgvector extension:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```
3. Get connection strings from Project Settings → Database:
   - **Direct connection**: For schema setup (DDL operations)
   - **Pooler connection**: For application runtime


### 3. Load Schema and Sample Data, Indexes in Supabase

### 4. Environment Configuration

Create `.env` file in project root:

# Specify the ENV VARIABLES in .env file using the .env-template
```
cp .env-template .env
```

### 5. Create Schema Descriptions (Optional)

Upload a YAML file to Supabase Storage with table/column descriptions:

```yaml
# schema_descriptions.yaml
tables:
  customer:
    description: "Customer information including name, email, and address"
  payment:
    description: "Payment transactions made by customers for rentals"
  film:
    description: "Movie information including title, description, and ratings"

columns:
  customer.email:
    description: "Customer's email address for communication"
  payment.amount:
    description: "Payment amount in dollars"
  film.rating:
    description: "MPAA rating (G, PG, PG-13, R, NC-17)"
```

### 6. Start the Server

```bash
# Development (with hot reload)
poetry run python scripts/run_dev.py

```

### 7. Index Schema

Before using NL2SQL, index your database schema:

```bash
curl -X POST http://<host>:<port>/api/v1/schema/index \
  -H "Content-Type: application/json" \
  -d '{"replace_existing": true}'
```

---

## Configuration

All configuration uses Pydantic Settings with environment variable support.
```
default host - localhost
default port - 8000
```

### Environment Variable Format

Use `__` (double underscore) as delimiter for nested config:

```env
DATABASE__DATABASE_URL=...     # DatabaseConfig.database_url
LLM__TEMPERATURE=0.1           # LLMConfig.temperature
NL2SQL__MAX_TABLES=6           # NL2SQLConfig.max_tables
```

---

## API Reference

Start the Server and Refer to the interactive **Swagger UI** documentation available at:

```
http://<host>:<port>/docs
```

This provides complete API documentation with request/response schemas, try-it-out functionality, and all available endpoints.

---

## Overview

NL2SQL converts natural language questions into safe, executable SQL queries and executes them to fetch final results. Unlike simple prompt-based approaches, this system uses a **multi-step pipeline** with:

- **Semantic Schema Retrieval**: Vector search to find relevant tables/columns/relationships
- **Deterministic Filtering**: Code-based filtering with hard caps (no LLM hallucination)
- **Safe SQL Generation**: LLM generates SQL with retry loop
- **Validation**: Static checks + EXPLAIN validation before execution
- **Read-Only Execution**: All queries run with read-only enforcement

### Key Features

✅ **Production Safety**: SELECT-only, read-only connections, SQL validation disabling data manipulation
✅ **Semantic Search**: pgvector-based schema retrieval  
✅ **Deterministic Control**: No autonomous agents, explicit orchestration  
✅ **Full Observability**: Structured JSON logging, trace IDs, provenance  
✅ **Modular Architecture**: Swappable components via dependency injection  


---

## References

This prototype draws inspiration from Swiggy's **Hermes Framework**—their production-grade Text-to-SQL system. The following blog posts from Swiggy Bytes provided valuable insights of the framework:

- [Hermes: A Text-to-SQL Solution at Swiggy](https://bytes.swiggy.com/hermes-a-text-to-sql-solution-at-swiggy-81573fb4fb6e)
- [Hermes V3: Building Swiggy's Conversational AI Analyst](https://bytes.swiggy.com/hermes-v3-building-swiggys-conversational-ai-analyst-a41057a2279d)



---

## Example Queries

The following examples demonstrate the system's ability to handle diverse query patterns. The SQL shown is the **actual output generated by the system** for each natural language query:

### 1. Geographic Filtering with Random Sampling

**Natural Language:**
> "List random customers from London at max 50"

**Generated SQL:**
```sql
SELECT c.customer_id, c.first_name, c.last_name, c.email, a.address, ci.city 
FROM customer c 
JOIN address a ON c.address_id = a.address_id 
JOIN city ci ON a.city_id = ci.city_id 
WHERE ci.city ILIKE 'London' 
ORDER BY RANDOM() 
LIMIT 50
```

**Demonstrates:**
- Case-insensitive filtering with `ILIKE`
- Multi-table joins through address → city
- Random ordering for sampling
- User-specified limit respected (50 vs default 100)

### 2. Business Analytics with Revenue Aggregation

**Natural Language:**
> "Which film categories generate the most revenue?"

**Generated SQL:**
```sql
SELECT c.name AS category, SUM(f.rental_rate) AS total_revenue 
FROM category c 
JOIN film_category fc ON c.category_id = fc.category_id 
JOIN film f ON fc.film_id = f.film_id 
GROUP BY c.name 
ORDER BY total_revenue DESC 
LIMIT 1000
```

**Demonstrates:**
- Business insight extraction
- Junction table traversal (film_category)
- SUM aggregation with GROUP BY
- Descending sort for "most/highest" queries

### 3. Complex Multi-Table Inventory Analysis

**Natural Language:**
> "How many copies of films in each category does each store have?"

**Generated SQL:**
```sql
SELECT s.store_id, fc.category_id, COUNT(i.inventory_id) AS copy_count 
FROM inventory i 
JOIN film_category fc ON i.film_id = fc.film_id 
JOIN store s ON i.store_id = s.store_id 
GROUP BY s.store_id, fc.category_id 
ORDER BY s.store_id, fc.category_id 
LIMIT 1000
```

**Demonstrates:**
- Multi-dimensional grouping (store × category)
- 3-table join chain
- COUNT aggregation for inventory tracking
- Ordered output for readability

### 4. Store Manager Lookup with String Concatenation

**Natural Language:**
> "Show me each store's city and its manager's name"

**Generated SQL:**
```sql
SELECT c.city, s.first_name || ' ' || s.last_name AS manager_name 
FROM store st 
JOIN address a ON st.address_id = a.address_id 
JOIN city c ON a.city_id = c.city_id 
JOIN staff s ON st.manager_staff_id = s.staff_id 
LIMIT 100
```

**Demonstrates:**
- String concatenation (`||`) for full name formatting
- 4-table join chain (store → address → city, store → staff)
- Multiple join paths from single table
- Entity relationship traversal (manager_staff_id → staff)

### 5. Customer Rental History with View Usage

**Natural Language:**
> "Give me all films names that customer john Smith has rented with the dates he rented them with oldest first"

**Generated SQL:**
```sql
SELECT title, rental_date 
FROM films_per_customer_rental 
WHERE LOWER(customer) = 'john smith' 
ORDER BY rental_date ASC 
LIMIT 100
```

**Demonstrates:**
- View utilization (films_per_customer_rental) for simplified queries
- Case-insensitive filtering with `LOWER()`
- Ascending date sort for "oldest first"
- Natural language temporal ordering understood

### 6. Aggregate Count with DISTINCT

**Natural Language:**
> "What is the total number of unique movies rented by all customers?"

**Generated SQL:**
```sql
SELECT COUNT(DISTINCT f.film_id) AS total_unique_movies 
FROM rental r 
JOIN inventory i ON r.inventory_id = i.inventory_id 
JOIN film f ON i.film_id = f.film_id
```

**Demonstrates:**
- `COUNT(DISTINCT ...)` for unique value aggregation
- Multi-table join chain (rental → inventory → film)
- No LIMIT needed for single-row aggregate results
- Implicit understanding of "unique" → DISTINCT


----

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FastAPI Application                            │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   /health   │  │/schema/index│  │/schema/search│ │   /nl2sql/query     │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
├─────────┴────────────────┴────────────────┴──────────────────────┴──────────┤
│                         Middleware (Trace ID, Logging)                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                              SERVICE LAYER                                  │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐  │
│  │   SchemaService     │  │   VectorService     │  │   NL2SQLService     │  │
│  │  (Schema metadata)  │  │  (Indexing/Search)  │  │  (Pipeline orch.)   │  │
│  └──────────┬──────────┘  └──────────┬──────────┘  └──────────┬──────────┘  │
├─────────────┴────────────────────────┴──────────────────────────┴───────────┤
│                            REPOSITORY LAYER                                 │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐        │
│  │ SchemaRepo   │ │ VectorRepo   │ │ SQLGenRepo   │ │ SQLValidRepo │        │
│  │(DB metadata) │ │(pgvector ops)│ │(LLM prompts) │ │(SQL checks)  │        │
│  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘ └──────┬───────┘        │
│         │                │                │                │                │
│  ┌──────┴───────┐ ┌──────┴───────┐ ┌──────┴───────┐ ┌──────┴───────┐        │
│  │ SQLExecRepo  │ │SchemaFilter  │ │              │ │              │        │
│  │(Read-only)   │ │(Deterministic)││              │ │              │        │
│  └──────┬───────┘ └──────────────┘ └──────────────┘ └──────────────┘        │
├─────────┴───────────────────────────────────────────────────────────────────┤
│                          INFRASTRUCTURE LAYER                               │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐           │
│  │  DatabaseClient  │  │    LLMClient     │  │ EmbeddingClient  │           │
│  │    (asyncpg)     │  │   (LangChain)    │  │   (LangChain)    │           │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘           │
│           │                     │                     │                     │
│  ┌────────┴─────────┐  ┌────────┴─────────┐  ┌────────┴─────────┐           │
│  │   PostgreSQL     │  │   OpenRouter     │  │   OpenRouter     │           │
│  │   + pgvector     │  │   (Claude/GPT)   │  │   (Embeddings)   │           │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘           │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Layer Responsibilities

| Layer | Responsibility | Example |
|-------|----------------|---------|
| **API** | HTTP handling, request/response models | Routes, middleware |
| **Service** | Business logic orchestration | NL2SQLService coordinates pipeline |
| **Repository** | Data access, external operations | VectorRepository, SQLGenerationRepository |
| **Infrastructure** | External system clients | DatabaseClient, LLMClient |
| **Domain** | Data models, errors, enums | Pydantic models, custom exceptions |

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Web Framework** | FastAPI | Async REST API with OpenAPI docs |
| **Database** | PostgreSQL + pgvector | Data storage + vector similarity search |
| **LLM Orchestration** | LangChain | LLM/embedding client abstraction |
| **LLM Provider** | OpenRouter | Access to Claude, GPT-4, etc. |
| **Embeddings** | OpenAI text-embedding-3-small | 1536-dim vectors via OpenRouter |
| **Configuration** | Pydantic Settings | Type-safe environment config |
| **Dependency Mgmt** | Poetry | Python package management |
| **Async DB** | asyncpg | High-performance PostgreSQL driver |
| **Storage** | Supabase Storage | YAML schema descriptions | Postgres Database

---

## Pipeline

The NL2SQL pipeline converts a natural language query to SQL through 6 deterministic steps:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         NL2SQL PIPELINE                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  User Query: "Show me the top 10 customers by total payments"               │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ STEP 1: SCHEMA RETRIEVAL (Vector Search)                            │    │
│  │                                                                     │    │
│  │  • Embed user query                                                 │    │
│  │  • Search pgvector for relevant columns (top_k=12)                  │    │
│  │  • Search pgvector for relevant relationships (top_k=12)            │    │
│  │  • Output: 24 candidate schema nodes with similarity scores         │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ STEP 2: SCHEMA FILTERING (Deterministic - No LLM)                   │    │
│  │                                                                     │    │
│  │  • Extract unique tables from columns                               │    │
│  │  • Apply hard caps: max_tables=6, max_columns=15, max_rels=6        │    │
│  │  • Add PK/FK columns for complete joins                             │    │
│  │  • Output: Filtered tables, columns, relationships                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ STEP 3: TABLE CONTEXT (Deterministic Lookup)                        │    │
│  │                                                                     │    │
│  │  • Fetch table descriptions by exact match (no embeddings)          │    │
│  │  • Add missing columns for tables with < 3 columns                  │    │
│  │  • Output: Tables with descriptions, complete column sets           │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ STEP 4: SQL GENERATION (LLM Call)                                   │    │
│  │                                                                     │    │
│  │  • Build prompt with filtered schema context                        │    │
│  │  • Call LLM (Claude/GPT) with JSON mode                             │    │
│  │  • Parse response: {success: bool, sql: string, reason: string}     │    │
│  │  • Output: Generated SQL or reason for failure                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ STEP 5: SQL VALIDATION (Static + EXPLAIN)                           │    │
│  │                                                                     │    │
│  │  • Check 1: Must be SELECT statement                                │    │
│  │  • Check 2: No dangerous keywords (INSERT, UPDATE, DELETE, DROP)    │    │
│  │  • Check 3: Only uses allowed tables from context                   │    │
│  │  • Check 4: EXPLAIN validation (syntax check)                       │    │
│  │  • On failure: Retry with error feedback (max 2 retries)            │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ STEP 6: SQL EXECUTION (Read-Only)                                   │    │
│  │                                                                     │    │
│  │  • Execute with read_only=True                                      │    │
│  │  • Enforce timeout (30s)                                            │    │
│  │  • Return rows with column names                                    │    │
│  │  • Output: Query results as JSON                                    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                              │
│                              ▼                                              │
│  Response: {sql, results, grounding, provenance, trace_id}                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Schema Embedding Strategy

Each schema element is stored as a separate vector document:

```
┌─────────────────────────────────────────────────────────────────┐
│ TABLE DOCUMENT                                                  │
├─────────────────────────────────────────────────────────────────┤
│ content: "Table: payment\nDescription: Customer payments..."    │
│ metadata: {node_type: "table", table_name: "payment", ...}      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ COLUMN DOCUMENT                                                 │
├─────────────────────────────────────────────────────────────────┤
│ content: "Column: payment.amount\nType: numeric..."             │
│ metadata: {node_type: "column", table_name: "payment",          │
│            column_name: "amount", sample_values: [...], ...}    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ RELATIONSHIP DOCUMENT                                           │
├─────────────────────────────────────────────────────────────────┤
│ content: "Relationship: payment.customer_id -> customer.id"     │
│ metadata: {node_type: "relationship", from_table: "payment",    │
│            from_column: "customer_id", to_table: "customer",...}│
└─────────────────────────────────────────────────────────────────┘
```

---

### Design Decisions

The following are **conscious architectural choices** made in this system:

#### 1. No Query Expansion - Grounded Source of Truth
- **Decision**: The user's natural language query is embedded directly without expansion, rewriting, or augmentation
- **Rationale**: Query expansion (adding synonyms, rephrasing) can introduce drift from user intent. We prefer a grounded approach where the retrieval is based solely on what the user actually asked
- **Trade-off**: May miss some relevant schema elements if user terminology differs from schema naming

#### 2. Maximize Recall Over Precision in Retrieval
- **Decision**: Retrieval step uses low/zero similarity thresholds and fetches more candidates than needed (top_k=12+)
- **Rationale**: It's better to retrieve potentially irrelevant schema elements than to miss relevant ones. Missing a critical table/column means the LLM cannot generate correct SQL
- **Trade-off**: More noise in retrieved context, but handled by next step

#### 3. Precision Handled via Deterministic Filtering
- **Decision**: After high-recall retrieval, a code-based filtering step applies hard caps (max_tables=6, max_columns=15)
- **Rationale**: Filtering is deterministic, predictable, and debuggable - no LLM hallucination risk. This separates "finding candidates" (recall) from "selecting relevant ones" (precision)
- **Trade-off**: Filtering logic must be maintained as schema complexity grows

#### 4. No Autonomous Agents - Explicit Orchestration
- **Decision**: Pipeline steps are explicitly coded, not driven by an autonomous agent deciding next actions
- **Rationale**: Agents are unpredictable and hard to debug. Explicit orchestration provides deterministic behavior, clear audit trails, and predictable latency
- **Trade-off**: Less flexible for complex multi-turn reasoning (but that's out of scope for V1)

#### 5. Single LLM Call for SQL Generation
- **Decision**: One LLM call with complete context, rather than multi-turn conversation or chain-of-thought agents
- **Rationale**: Reduces latency and unpredictability. Retry loop handles failures without agent complexity
- **Trade-off**: Complex queries may need better prompting and multi-step reasoning

#### 6. Schema Descriptions in Embeddings, Sample Values in Metadata
- **Decision**: Embed semantic descriptions for search, store sample values as metadata (not embedded)
- **Rationale**: Descriptions improve semantic matching ("customer email" matches "contact info"). Sample values help LLM understand data patterns but shouldn't affect retrieval ranking
- **Trade-off**: Requires manual or AI-assisted description authoring

#### 7. Validation Before Execution - Never Trust LLM Output
- **Decision**: All generated SQL passes through static checks + EXPLAIN validation before execution
- **Rationale**: LLMs can hallucinate tables, generate unsafe SQL, or produce syntax errors. Defense in depth ensures safety even if prompt engineering fails
- **Trade-off**: Additional latency for validation step

#### 8. Deterministic Join Completion (Junction Tables)
- **Decision**: Join paths are completed deterministically using schema relationships (e.g., junction tables), rather than inferred by the LLM
- **Rationale**: Users rarely mention junction tables, but SQL correctness depends on them. Structural join completion prevents hallucinated joins and ensures correctness
- **Trade-off**: Requires schema graph analysis and limits join depth (e.g., one-hop only)


---


## Security

### SQL Injection Prevention

1. **SELECT-only**: All queries must start with `SELECT`
2. **Keyword blocking**: INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE blocked
3. **Table allowlist**: Only tables in retrieved context can be used
4. **EXPLAIN validation**: Syntax checked before execution
5. **Read-only execution**: Connections use `SET TRANSACTION READ ONLY`

### Prompt Injection Mitigation

1. **Schema context limit**: Only top-K relevant schema elements sent to LLM
2. **Hard caps**: max_tables, max_columns, max_relationships enforced
3. **Validation loop**: LLM output validated before execution
4. **No direct execution**: SQL always passes through validation layer

----

## License

MIT License - see LICENSE file for details.

---

