import logging
import json
import time
from db.session import SessionLocal
from agent.router import process_pipeline

# Configure clean console logging for test output
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("helpdesk_test")

TEST_QUERIES = [
    # 1. Direct LLM Response Path
    "Hello! Who are you and how can you help me today?",
    
    # 2. Direct LLM Response Path (General Tech Knowledge)
    "Explain what a DNS server does in simple terms.",
    
    # 3. RAG Retrieval Path
    "How do I deploy the backend IT service?",
    
    # 4. RAG Retrieval Path
    "What is the SSID and security configuration for the office secure Wi-Fi?",
    
    # 5. DB Query Path
    "How many active users are currently registered in the database?",
    
    # 6. DB Query Path
    "List the usernames of all suspended accounts in the system.",
    
    # 7. Tool Execution Path (Calculator)
    "What is the result of (1200 + 450) * 1.13?",
    
    # 8. Tool Execution Path (File Search)
    "Search local file repositories for 'wireguard' network configs.",
    
    # 9. Multi-source Coordination Path (DB users role count + RAG deployment steps)
    "How many users with the role 'developer' are in the system, and what is the exact Kubernetes command to deploy the backend service?",
    
    # 10. DB Query Path (Self-monitoring query statistics)
    "What is the average latency of the queries logged in the system so far?"
]

def run_diagnostic_suite():
    print("=" * 100)
    print("      IT HELPDESK & KNOWLEDGE ASSISTANT -- DIAGNOSTIC & EVALUATION SUITE      ")
    print("=" * 100)
    
    db = SessionLocal()
    try:
        total_latency = 0.0
        success_count = 0
        
        for idx, query in enumerate(TEST_QUERIES, 1):
            print(f"\n[{idx}/10] Query: '{query}'")
            print("-" * 50)
            
            # Run query pipeline
            result = process_pipeline(db, query)
            
            print(f">> Explicit Routing Decision: {result['routing_decision']}")
            print(f">> Latency:                   {result['latency_seconds']:.4f} seconds")
            
            if result['tools_used']:
                print(f">> Tools/SQL Used:            {json.dumps(result['tools_used'], default=str)}")
                
            if result['retrieved_documents']:
                print(f">> Retrieved Chunks Count:    {len(result['retrieved_documents'])}")
                for chunk in result['retrieved_documents']:
                    print(f"  - Chunk (Similarity: {chunk['similarity']:.4f}): \"{chunk['content'][:80]}...\"")
                    
            print(f">> Synthesized Response:\n{result['response']}")
            print("=" * 100)
            
            total_latency += result['latency_seconds']
            success_count += 1
            
            # Add pacing delay to respect free-tier rate limits
            if idx < len(TEST_QUERIES):
                print(f"\n[Pacing Delay] Waiting 4 seconds before next query to respect API quota limits...")
                time.sleep(4.0)
            
        avg_latency = total_latency / success_count if success_count else 0.0
        print(f"\nEvaluation Suite Finished Successfully!")
        print(f"Total Queries Processed: {success_count}/10")
        print(f"Average Latency:         {avg_latency:.4f} seconds")
        print("=" * 100)
        
    except Exception as e:
        print(f"Error during diagnostic suite execution: {e}")
        raise e
    finally:
        db.close()

if __name__ == "__main__":
    run_diagnostic_suite()
