"""
End-to-End Environment Validation Test

Run this script to verify all environment variables and connections are working:

    python test_env.py

Or with verbose output:

    python test_env.py --verbose
"""

import os
import sys
import argparse
from typing import Dict, List, Tuple


def print_header(text: str):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")


def print_success(text: str):
    print(f"  ✓ {text}")


def print_error(text: str):
    print(f"  ✗ {text}")


def print_warning(text: str):
    print(f"  ⚠ {text}")


def print_info(text: str):
    print(f"  ℹ {text}")


class EnvTester:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.success: List[str] = []
        
    def test_env_file(self) -> bool:
        """Check if .env file exists"""
        print_header("1. Environment File Check")
        
        env_exists = os.path.exists(".env")
        env_example_exists = os.path.exists(".env.example")
        
        if env_exists:
            print_success(".env file found")
            self.success.append(".env file exists")
        else:
            print_error(".env file NOT found")
            print_info("Run: cp .env.example .env")
            self.errors.append(".env file missing")
            
        if env_example_exists:
            print_success(".env.example template found")
        else:
            print_warning(".env.example not found")
            
        return env_exists
    
    def test_required_vars(self) -> bool:
        """Test required environment variables"""
        print_header("2. Required Environment Variables")
        
        # Load .env if exists
        try:
            from dotenv import load_dotenv
            load_dotenv()
            print_success("Loaded .env file")
        except ImportError:
            print_warning("python-dotenv not installed, using system env vars only")
        
        required_vars = [
            ("ENV", "LOCAL or CLOUD"),
        ]
        
        # At least one LLM provider
        llm_providers = [
            ("OPENAI_API_KEY", "OpenAI"),
            ("GOOGLE_API_KEY", "Google/Gemini"),
            ("GROQ_API_KEY", "Groq"),
            ("COHERE_API_KEY", "Cohere"),
        ]
        
        # Check required vars
        all_present = True
        for var, desc in required_vars:
            value = os.getenv(var)
            if value:
                print_success(f"{var}: {desc} = {value[:10]}..." if len(str(value)) > 10 else f"{var}: {desc} = {value}")
                self.success.append(f"{var} is set")
            else:
                print_error(f"{var}: {desc} - NOT SET")
                self.errors.append(f"{var} is missing")
                all_present = False
        
        # Check LLM providers (at least one needed)
        print_info("\n  LLM Providers (at least one required):")
        llm_found = False
        for var, provider in llm_providers:
            value = os.getenv(var)
            if value:
                masked = value[:10] + "..." + value[-4:] if len(value) > 20 else "***"
                print_success(f"  {provider} ({var}): {masked}")
                llm_found = True
                self.success.append(f"{provider} API key set")
            else:
                print_info(f"  {provider} ({var}): not set")
        
        if not llm_found:
            print_error("\n  No LLM provider API key found!")
            print_info("  Set at least one: OPENAI_API_KEY, GOOGLE_API_KEY, or GROQ_API_KEY")
            self.errors.append("No LLM provider configured")
            all_present = False
        else:
            print_success("\n  At least one LLM provider configured")
            
        return all_present
    
    def test_qdrant_config(self) -> bool:
        """Test Qdrant configuration"""
        print_header("3. Qdrant Vector Database")
        
        env_mode = os.getenv("ENV", "LOCAL").upper()
        
        if env_mode == "LOCAL":
            qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
            print_info(f"Mode: LOCAL")
            print_info(f"QDRANT_URL: {qdrant_url}")
            
            # Try to connect
            try:
                from qdrant_client import QdrantClient
                client = QdrantClient(url=qdrant_url)
                collections = client.get_collections()
                print_success(f"Connected to Qdrant at {qdrant_url}")
                print_info(f"  Existing collections: {[c.name for c in collections.collections]}")
                self.success.append("Qdrant local connection OK")
                return True
            except Exception as e:
                print_error(f"Cannot connect to Qdrant: {e}")
                print_info("  Make sure Qdrant is running:")
                print_info("  docker run -p 6333:6333 qdrant/qdrant")
                self.errors.append("Qdrant connection failed")
                return False
                
        else:  # CLOUD mode
            cloud_url = os.getenv("QDRANT_CLOUD_URL")
            cloud_key = os.getenv("QDRANT_CLOUD_API_KEY")
            
            print_info(f"Mode: CLOUD")
            
            if not cloud_url or not cloud_key:
                print_error("QDRANT_CLOUD_URL or QDRANT_CLOUD_API_KEY not set")
                self.errors.append("Qdrant cloud credentials missing")
                return False
            
            try:
                from qdrant_client import QdrantClient
                client = QdrantClient(url=cloud_url, api_key=cloud_key)
                collections = client.get_collections()
                print_success(f"Connected to Qdrant Cloud")
                print_info(f"  URL: {cloud_url[:50]}...")
                print_info(f"  Collections: {[c.name for c in collections.collections]}")
                self.success.append("Qdrant cloud connection OK")
                return True
            except Exception as e:
                print_error(f"Cannot connect to Qdrant Cloud: {e}")
                self.errors.append("Qdrant cloud connection failed")
                return False
    
    def test_embedding_providers(self) -> bool:
        """Test embedding provider connectivity"""
        print_header("4. Embedding Providers")
        
        all_ok = True
        
        # Test Google/Gemini
        google_key = os.getenv("GOOGLE_API_KEY")
        if google_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=google_key)
                # Try a simple embedding
                result = genai.embed_content(
                    model="models/gemini-embedding-001",
                    content="test",
                    task_type="retrieval_document"
                )
                print_success("Google Gemini embedding: OK")
                self.success.append("Gemini embedding works")
            except Exception as e:
                print_error(f"Google Gemini embedding failed: {e}")
                self.warnings.append("Gemini embedding issue")
                all_ok = False
        else:
            print_info("Google API key not set, skipping Gemini test")
        
        # Test OpenAI
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            try:
                import openai
                client = openai.OpenAI(api_key=openai_key)
                # Test with a simple embedding
                response = client.embeddings.create(
                    model="text-embedding-3-small",
                    input="test"
                )
                print_success("OpenAI embedding: OK")
                self.success.append("OpenAI embedding works")
            except Exception as e:
                print_error(f"OpenAI embedding failed: {e}")
                self.warnings.append("OpenAI embedding issue")
                all_ok = False
        else:
            print_info("OpenAI API key not set, skipping OpenAI test")
        
        return all_ok
    
    def test_llm_providers(self) -> bool:
        """Test LLM provider connectivity"""
        print_header("5. LLM Providers")
        
        all_ok = True
        
        # Test Groq
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            try:
                from groq import Groq
                client = Groq(api_key=groq_key)
                # Simple completion test
                response = client.chat.completions.create(
                    model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
                    messages=[{"role": "user", "content": "Hi"}],
                    max_tokens=10
                )
                print_success(f"Groq LLM: OK (model: {os.getenv('GROQ_MODEL', 'llama-3.1-8b-instant')})")
                self.success.append("Groq LLM works")
            except Exception as e:
                print_error(f"Groq LLM failed: {e}")
                self.warnings.append("Groq LLM issue")
                all_ok = False
        else:
            print_info("Groq API key not set, skipping")
        
        # Test OpenAI
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            try:
                import openai
                client = openai.OpenAI(api_key=openai_key)
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "Hi"}],
                    max_tokens=10
                )
                print_success("OpenAI LLM: OK")
                self.success.append("OpenAI LLM works")
            except Exception as e:
                print_error(f"OpenAI LLM failed: {e}")
                self.warnings.append("OpenAI LLM issue")
                all_ok = False
        else:
            print_info("OpenAI API key not set, skipping")
        
        # Test Gemini
        google_key = os.getenv("GOOGLE_API_KEY")
        if google_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=google_key)
                model = genai.GenerativeModel(os.getenv("GEMINI_CHAT_MODEL", "gemini-2.0-flash"))
                response = model.generate_content("Hi")
                print_success(f"Gemini LLM: OK (model: {os.getenv('GEMINI_CHAT_MODEL', 'gemini-2.0-flash')})")
                self.success.append("Gemini LLM works")
            except Exception as e:
                print_error(f"Gemini LLM failed: {e}")
                self.warnings.append("Gemini LLM issue")
                all_ok = False
        else:
            print_info("Google API key not set, skipping")
        
        return all_ok
    
    def test_supabase(self) -> bool:
        """Test Supabase connection if configured"""
        print_header("6. Supabase Database (Optional)")
        
        supabase_url = os.getenv("SUPABASE_URL")
        
        if not supabase_url:
            print_info("SUPABASE_URL not set, skipping database test")
            return True
        
        try:
            # Just verify URL format, don't actually connect
            if supabase_url.startswith("postgresql://"):
                print_success("Supabase URL format looks valid")
                masked = supabase_url.replace(supabase_url.split(":")[2].split("@")[0], "***")
                print_info(f"  URL: {masked}")
                self.success.append("Supabase URL configured")
                return True
            else:
                print_warning("Supabase URL doesn't look like a PostgreSQL URL")
                return False
        except Exception as e:
            print_warning(f"Supabase URL check failed: {e}")
            return False
    
    def test_imports(self) -> bool:
        """Test that all required packages are installed"""
        print_header("7. Package Dependencies")
        
        required_packages = [
            "fastapi",
            "uvicorn",
            "pydantic",
            "pydantic_settings",
            "qdrant_client",
            "openai",
            "google.generativeai",
            "groq",
            "tiktoken",
            "PyPDF2",
            "pdfplumber",
        ]
        
        all_ok = True
        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
                print_success(f"{package}: installed")
            except ImportError:
                print_error(f"{package}: NOT installed")
                print_info(f"  Run: pip install {package}")
                self.errors.append(f"Package missing: {package}")
                all_ok = False
        
        return all_ok
    
    def test_directories(self) -> bool:
        """Test that required directories exist"""
        print_header("8. Directory Structure")
        
        required_dirs = [
            "app",
            "app/api",
            "app/services",
            "app/models",
            "app/utils",
            "uploads",
        ]
        
        all_ok = True
        for dir_path in required_dirs:
            if os.path.isdir(dir_path):
                print_success(f"{dir_path}/: exists")
            else:
                print_error(f"{dir_path}/: NOT FOUND")
                print_info(f"  Create with: mkdir -p {dir_path}")
                self.errors.append(f"Directory missing: {dir_path}")
                all_ok = False
        
        return all_ok
    
    def run_all_tests(self) -> Tuple[int, int, int]:
        """Run all tests and return summary"""
        print_header("RAG PIPELINE BACKEND - ENVIRONMENT TEST")
        print(f"  Running tests...\n")
        
        tests = [
            self.test_env_file,
            self.test_required_vars,
            self.test_qdrant_config,
            self.test_embedding_providers,
            self.test_llm_providers,
            self.test_supabase,
            self.test_imports,
            self.test_directories,
        ]
        
        for test in tests:
            try:
                test()
            except Exception as e:
                print_error(f"Test failed with exception: {e}")
                self.errors.append(f"Test exception: {e}")
        
        return len(self.errors), len(self.warnings), len(self.success)
    
    def print_summary(self):
        """Print final summary"""
        print_header("TEST SUMMARY")
        
        total_errors = len(self.errors)
        total_warnings = len(self.warnings)
        total_success = len(self.success)
        
        if total_errors == 0:
            print_success(f"All critical checks passed! ({total_success} success)")
            if total_warnings > 0:
                print_warning(f"{total_warnings} non-critical warnings")
        else:
            print_error(f"{total_errors} critical errors found")
            print_warning(f"{total_warnings} warnings")
            print_success(f"{total_success} passed")
        
        if self.verbose:
            print("\n  Detailed Results:")
            print(f"    Errors: {total_errors}")
            print(f"    Warnings: {total_warnings}")
            print(f"    Success: {total_success}")
            
            if self.errors:
                print("\n  Errors:")
                for e in self.errors:
                    print(f"    - {e}")
            
            if self.warnings:
                print("\n  Warnings:")
                for w in self.warnings:
                    print(f"    - {w}")
        
        print(f"\n{'='*60}\n")
        
        return 0 if total_errors == 0 else 1


def main():
    parser = argparse.ArgumentParser(description="Test RAG backend environment")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    args = parser.parse_args()
    
    tester = EnvTester(verbose=args.verbose)
    tester.run_all_tests()
    exit_code = tester.print_summary()
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
