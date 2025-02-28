import json
import os
import boto3
import pandas as pd
from tqdm import tqdm
from flask import Flask, request, jsonify
from botocore.client import Config
from langchain_aws import BedrockEmbeddings
from elasticsearch import Elasticsearch
from llama_index.llms.bedrock import Bedrock
from llama_index.core import Settings

app = Flask(__name__)

# Initialize AWS clients and models
class BedrockClient:
    def __init__(self):
        self.bedrock_config = Config(connect_timeout=120, read_timeout=120, retries={'max_attempts': 0})
        self.bedrock_client = boto3.client(
            'bedrock-runtime',
            region_name='us-west-2',
            aws_access_key_id='AKIAUUN3K6A2Y7E3NQLB',
            aws_secret_access_key='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',
        )
        
        self.model_id = 'anthropic.claude-3-5-sonnet-20241022-v2:0'
        self.embedding_model_id = 'amazon.titan-embed-text-v2:0'
        
        self.llm = Bedrock(model=self.model_id, client=self.bedrock_client)
        self.embeddings = BedrockEmbeddings(model_id=self.embedding_model_id, client=self.bedrock_client)
        
        # Configure the global embedding model for Settings
        Settings.embed_model = self.embeddings

    def get_llm(self):
        return self.llm
    
    def get_embeddings(self):
        return self.embeddings


# Elasticsearch client
class ElasticsearchClient:
    def __init__(self):
        self.cloud_id = "DBS:dXMtd2VzdC0yLmF3cy5lbGFzdGljLmNsb3VkJGExOWFlMjYzZWFhMzRlYTY4NTAxZGI3NWJjYzQ1NTQyLmVzJGExOWFlMjYzZWFhMzRlYTY4NTAxZGI3NWJjYzQ1NTQy"
        self.cloud_api_key = "SlBnRVNwVUI4aUVtRmU1ZU9SRDU6YlA3ejQ1NlUyZlUzeW5OdGRhOVZJUQ=="
        self.index_name = "sms"
        self.es_endpoint = "https://dbs-a19ae2.es.us-west-2.aws.elastic.cloud:443"
        
        try:
            self.es_client = Elasticsearch(
                hosts=[self.es_endpoint],
                api_key=self.cloud_api_key
            )
        except Exception as e:
            print(f"Error initializing Elasticsearch client: {str(e)}")
            self.es_client = None

    def search(self, query_embedding):
        es_query = {
            "query": {
                "script_score": {
                    "query": {
                        "match_all": {}
                    },
                    "script": {
                        "source": "return (cosineSimilarity(params.query_vector, 'embedding') + 1)",
                        "params": {
                            "query_vector": query_embedding
                        }
                    }
                }
            },
            "_source": ["v1", "v2"],
            "size": 10
        }
        
        try:
            response = self.es_client.search(index=self.index_name, body=es_query)
            return response
        except Exception as e:
            print(f"Error searching Elasticsearch: {str(e)}")
            return {"hits": {"hits": []}}


# Data processor for creating embeddings
class DataProcessor:
    def __init__(self, bedrock_client):
        self.bedrock_client = bedrock_client
    
    def process_csv_with_embeddings(self, csv_path, output_path, limit=500):
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'ISO-8859-1', 'cp1252']
        
        # Try each encoding until one works
        for encoding in encodings:
            try:
                # Read the CSV file with the specified encoding
                df = pd.read_csv(csv_path, encoding=encoding)
                print(f"Successfully read the CSV with {encoding} encoding")
                break
            except UnicodeDecodeError:
                print(f"Failed to read with {encoding} encoding, trying next...")
                if encoding == encodings[-1]:
                    raise Exception("Could not read the CSV file with any of the attempted encodings")
        
        df = df.head(limit)
        
        # Add a new column for embeddings
        df['embedding'] = None
        
        # Process each row to get embeddings for the v2 column
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            # Get the text from v2 column
            text = row['v2']
            
            if isinstance(text, str):
                # Generate embedding using the provided function
                embedding = Settings.embed_model.get_query_embedding(text)
                
                # Store the embedding in the new column
                df.at[idx, 'embedding'] = embedding
            else:
                print(f"Warning: Non-string value found in row {idx}, v2 column: {text}")
        
        # Reorder columns to put embedding as the third column
        column_order = df.columns.tolist()
        column_order.remove('embedding')
        column_order.insert(2, 'embedding')
        df = df[column_order]
        
        # Write the processed DataFrame to a new CSV file with the same encoding
        df.to_csv(output_path, index=False, encoding=encoding)
        
        return {"status": "success", "message": f"Processed {len(df)} rows and saved to {output_path}"}


# RAG processor
class RAGProcessor:
    def __init__(self, bedrock_client, es_client):
        self.bedrock_client = bedrock_client
        self.es_client = es_client
        self.llm = bedrock_client.get_llm()
    
    def get_context_document_str(self, elasticsearch_results):
        context_entries = []
        
        for hit in elasticsearch_results['hits']['hits']:
            source = hit['_source']
            classification = source.get('v1', '')  # 'ham' or 'spam'
            message = source.get('v2', '')         # The actual message content
            
            # Create a formatted entry for each document
            formatted_entry = f"Classification: {classification}\nMessage: {message}"
            context_entries.append(formatted_entry)
        
        # Join all entries with clear separation
        context_str = "\n\n".join([f"Document {i+1}:\n{entry}" for i, entry in enumerate(context_entries)])
        
        return context_str
    
    def create_context_prompt(self, context_str, user_query):
        prompt = f"""You are an expert system for detecting spam, scam, and fraudulent messages. Your task is to analyze the provided message and determine if it is legitimate ("ham") or spam/scam ("spam").

REFERENCE CONTEXT:
Below are relevant context documents retrieved from our database. These are messages with known classifications that are most similar to the current user message. Use these context documents to inform your judgment.

{context_str}

USER MESSAGE TO ANALYZE:
{user_query}

INSTRUCTIONS:
1. Carefully analyze the user message in comparison with the provided context documents.
2. Pay special attention to the classifications of similar messages in the context.
3. Consider common spam indicators including:
   - Urgency or pressure tactics
   - Requests for personal information
   - Unexpected money offers or requests
   - Suspicious links or attachments
   - Grammatical errors and unusual formatting
   - Impersonation of organizations or individuals
   - Too-good-to-be-true offers
4. Make your determination based primarily on the context documents, using them as reference points.
5. If the user message closely resembles messages classified as "spam" in the context, this is strong evidence for a "spam" classification.
6. Similarly, if it resembles "ham" messages, this supports a "ham" classification.

OUTPUT FORMAT:
Provide your analysis in the following JSON-like format:
"Judgment": "spam" or "ham",
"Reasoning": Your detailed explanation for the judgment

Your reasoning should include specific elements from the message that led to your conclusion and reference relevant patterns from the context documents.
"""
        return prompt
    
    def process_message(self, user_message):
        # Generate embedding for the user message
        query_embedding = Settings.embed_model.get_query_embedding(user_message)
        
        # Get similar messages from Elasticsearch
        es_response = self.es_client.search(query_embedding)
        
        # Get context string from search results
        context_str = self.get_context_document_str(es_response)
        
        # Create prompt with context and user message
        input_prompt = self.create_context_prompt(context_str, user_message)
        
        # Get response from LLM
        answer = self.llm.complete(input_prompt)
        
        # Parse and return result
        try:
            result_text = json.loads(answer.model_dump_json())['text']
            # Try to parse the result as JSON if possible
            try:
                result_json = json.loads(result_text)
                return result_json
            except:
                # If not valid JSON, return as text
                return {"raw_result": result_text}
        except Exception as e:
            return {"error": str(e), "raw_result": str(answer)}


# Initialize clients
bedrock_client = BedrockClient()
es_client = ElasticsearchClient()
data_processor = DataProcessor(bedrock_client)
rag_processor = RAGProcessor(bedrock_client, es_client)


# Define Flask routes
@app.route('/process_csv', methods=['POST'])
def process_csv():
    try:
        data = request.json
        input_csv = data.get('input_csv')
        output_csv = data.get('output_csv')
        limit = data.get('limit', 500)
        
        if not input_csv or not output_csv:
            return jsonify({"error": "Both input_csv and output_csv paths are required"}), 400
        
        result = data_processor.process_csv_with_embeddings(input_csv, output_csv, limit)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/analyze_message', methods=['POST'])
def analyze_message():
    try:
        data = request.json
        user_message = data.get('message')
        
        if not user_message:
            return jsonify({"error": "Message is required"}), 400
        
        result = rag_processor.process_message(user_message)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)