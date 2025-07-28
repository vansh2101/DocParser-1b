
# 📘 Round 1B – Modular Document Understanding Pipeline (Ollama Edition)

A fully modular, offline-compatible document understanding pipeline using:
- **YOLO-based layout detection**
- **Tesseract OCR**
- **Ollama-powered text summarization and topic generation (Qwen 0.5B)**
- **Intelligent section matching**

Built with performance, configurability, and multi-document support in mind.

---

## 🧠 Objective

Given:
- A persona (e.g., HR professional)
- A job-to-be-done (e.g., "Create and manage fillable forms...")
- A list of PDFs

→ **Extract ranked, relevant document sections** aligned with the task.

---

## 🧩 Modular Components

```bash
docParserv2/
├── main1B.js              
├── ollama-handler.js      
├── pdf-processor.js       
├── document-analyzer.js  
├── semantic-matcher.js    
├── matching_utils.js      
├── test-pipeline.js      
````

---

---

## 📥 Input Format

`challenge1b_input.json`

```json
{
  "challenge_info": {
    "challenge_id": "round_1b_003",
    "test_case_name": "create_manageable_forms",
    "description": "Creating manageable forms"
  },
  "persona": {
    "role": "HR professional"
  },
  "job_to_be_done": {
    "task": "Create and manage fillable forms for onboarding and compliance."
  },
  "documents": [
    {
      "filename": "document.pdf",
      "title": "Form Design Guide"
    }
  ]
}
```

---


## 🧠 Key Features

| Feature                  | Description                               |
| ------------------------ | ----------------------------------------- |
| 🔍 **Topic Ranking**     | Uses Qwen 0.5B via Ollama                 |
| 🧠 **AI Matching**       | Combines fuzzy, cosine, and Ollama scores |
| 🧾 **Layout-Aware**      | Handles multi-column, vertical PDFs       |
| 📦 **Modular**           | Replace any module independently          |
| 🪄 **Summarizer**        | Summarizes each PDF using LLM             |
| 🧱 **Structure Builder** | Hierarchical JSON per document            |
| 💡 **Offline Ready**     | No cloud or API usage                     |

---

## 🧪 Running the Pipeline

### 🔧 Full Pipeline (main1B.js)

```bash
node main1B.js
```

### 🧪 Test Components (test-pipeline.js)

```bash
node test-pipeline.js
```

---

## 🛠️ Custom Configuration

### Example

```js
const customConfig = {
  ollama: {
    modelName: "gbenson/qwen2.5-0.5b-instruct",
    maxRetries: 5,
    timeoutMs: 60000,
  },
  matching: {
    fuzzyWeight: 0.3,
    cosineWeight: 0.4,
    aiWeight: 0.3,
    minMatchScore: 0.2,
    maxConcurrentAIRequests: 3
  },
  processing: {
    confidenceThreshold: 0.6,
    minTextLength: 20
  }
};

const pipeline = new Round1BPipeline(customConfig);
await pipeline.run();
```

---

## 📤 Output Structure

### ✅ `output.json`

```json
{
  "metadata": {
    "input_documents": ["document.pdf"],
    "persona": "HR professional",
    "job_to_be_done": "Create and manage fillable forms...",
    "processing_time_seconds": 32.7,
    "pipeline_version": "modular-v2.0",
    "ai_model": "gbenson/qwen2.5-0.5b-instruct"
  },
  "ranked_topics": [
    "Form Creation",
    "Compliance Requirements",
    "Data Collection Fields"
  ],
  "extracted_sections": [
    {
      "document": "document.pdf",
      "topic": "Form Creation",
      "section_title": "Create Fillable PDFs",
      "page_number": 3,
      "importance_rank": 1,
      "match_score": 0.87,
      "match_type": "section-header"
    }
  ],
  "subsection_analysis": [
    {
      "document": "document.pdf",
      "refined_text": "To create interactive PDF forms, use the Prepare Form tool...",
      "page_number": 3
    }
  ]
}
```

---

## 🧩 Advanced Matching Logic

```js
score = fuzzyMatch(topic, heading) * 0.3
      + cosineSimilarity(topic, heading) * 0.4
      + ollamaSemanticScore(topic, heading) * 0.3;
```



