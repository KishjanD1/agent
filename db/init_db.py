import logging
from sqlalchemy import text
from db.session import Base, engine, SessionLocal
from db.models import User, Document, DocumentChunk, QueryLog
from rag.pipeline import ingest_document

logger = logging.getLogger("helpdesk_assistant.init_db")

def seed_data(db):
    logger.info("Checking if database already has seeded data...")
    user_count = db.query(User).count()
    doc_count = db.query(Document).count()
    
    if user_count > 0 or doc_count > 0:
        logger.info("Database already seeded. Skipping data seeding.")
        return
        
    logger.info("Seeding users table...")
    # Seed mock user data
    mock_users = [
        User(username="alice_admin", email="alice@company.com", status="active", role="admin"),
        User(username="bob_dev", email="bob@company.com", status="active", role="developer"),
        User(username="charlie_user", email="charlie@company.com", status="active", role="user"),
        User(username="diana_user", email="diana@company.com", status="active", role="user"),
        User(username="evan_suspended", email="evan@company.com", status="suspended", role="user"),
        User(username="fiona_dev", email="fiona@company.com", status="active", role="developer"),
        User(username="george_offline", email="george@company.com", status="offline", role="user"),
        User(username="hannah_user", email="hannah@company.com", status="active", role="user"),
        User(username="ian_dev", email="ian@company.com", status="active", role="developer"),
        User(username="julia_suspended", email="julia@company.com", status="suspended", role="user"),
    ]
    db.add_all(mock_users)
    db.commit()
    logger.info(f"Seeded {len(mock_users)} mock users.")
    
    logger.info("Seeding corporate IT helpdesk documents in the RAG pipeline...")
    # Seed mock corporate documentation
    it_docs = [
        (
            "To deploy the backend IT service, make sure the git branch is merged into 'main'. "
            "The CI pipeline will compile the Docker container and run pytest. "
            "Deploy the container via Kubernetes using the command: 'kubectl rollout restart deployment/backend-service -n it-helpdesk'. "
            "Alembic database migrations ('alembic upgrade head') are executed automatically during the pod initialization phase."
        , {"category": "deployment", "title": "Backend Service Deployment Guide"}),
        (
            "IT Corporate Network Configuration and Wifi SSID Setup:\n"
            "Office Secure Wi-Fi SSID is 'IT-Corp-Secure'. "
            "Wi-Fi Security authentication requires WPA3-Enterprise (PEAP/MSCHAPv2) with corporate LDAP credentials. "
            "Local VPN Gateway is vpn.internal-corp.net. VPN protocols supported are OpenVPN (Port 1194 UDP) and WireGuard (Port 51820 UDP)."
        , {"category": "network", "title": "Corporate Wifi and VPN Setup"}),
        (
            "Office Printer Setup and Installation Instructions:\n"
            "To add the corporate high-capacity printer on Windows, open Settings -> Bluetooth & Devices -> Printers & Scanners. "
            "Click 'Add device', then select 'The printer that I want isn't listed'. "
            "Select 'Add a printer using an IP address or hostname'. Enter Device Type: 'TCP/IP Device' and Hostname/IP: '192.168.4.150'. "
            "Choose Driver: 'HP LaserJet Pro M404-M405' (Standard PCL6 driver) and print a test page to verify."
        , {"category": "hardware", "title": "Office Printer Installation Guide"}),
        (
            "Microsoft 365 Exchange Online Corporate Email Settings:\n"
            "IMAP server is 'outlook.office365.com' on Port 993 (SSL/TLS). "
            "SMTP server is 'smtp.office365.com' on Port 587 (STARTTLS). "
            "Multi-factor authentication (MFA) via Microsoft Authenticator is mandatory for all email accounts."
        , {"category": "email", "title": "M365 Corporate Email Configurations"})
    ]
    
    for content, metadata in it_docs:
        ingest_document(db, content, metadata)
        
    logger.info("Successfully seeded helpdesk documents.")

def init_database():
    logger.info("Initializing PostgreSQL schema and tables...")
    
    # Drop all tables first for a clean seed
    Base.metadata.drop_all(bind=engine)
    logger.info("Dropped existing tables.")
    
    # Recreate all tables
    Base.metadata.create_all(bind=engine)
    logger.info("SQLAlchemy tables created successfully.")
    
    db = SessionLocal()
    try:
        if engine.dialect.name != "sqlite":
            # Create dot product mathematical helper function
            logger.info("Creating custom PL/pgSQL dot_product function...")
            db.execute(text("""
            CREATE OR REPLACE FUNCTION dot_product(a double precision[], b double precision[])
            RETURNS double precision AS $$
            DECLARE
              s double precision := 0;
              i integer;
              len_a integer;
              len_b integer;
            BEGIN
              len_a := cardinality(a);
              len_b := cardinality(b);
              IF len_a IS NULL OR len_b IS NULL OR len_a <> len_b THEN
                RETURN 0;
              END IF;
              FOR i IN 1..len_a LOOP
                s := s + a[i] * b[i];
              END LOOP;
              RETURN s;
            END;
            $$ LANGUAGE plpgsql IMMUTABLE;
            """))
            
            # Create vector magnitude mathematical helper function
            logger.info("Creating custom PL/pgSQL magnitude function...")
            db.execute(text("""
            CREATE OR REPLACE FUNCTION magnitude(a double precision[])
            RETURNS double precision AS $$
            DECLARE
              s double precision := 0;
              i integer;
              len integer;
            BEGIN
              len := cardinality(a);
              IF len IS NULL THEN
                RETURN 0;
              END IF;
              FOR i IN 1..len LOOP
                s := s + a[i] * a[i];
              END LOOP;
              RETURN sqrt(s);
            END;
            $$ LANGUAGE plpgsql IMMUTABLE;
            """))
            
            # Create cosine similarity function combining both functions
            logger.info("Creating custom PL/pgSQL cosine_similarity function...")
            db.execute(text("""
            CREATE OR REPLACE FUNCTION cosine_similarity(a double precision[], b double precision[])
            RETURNS double precision AS $$
            DECLARE
              dp double precision;
              mag_a double precision;
              mag_b double precision;
            BEGIN
              dp := dot_product(a, b);
              mag_a := magnitude(a);
              mag_b := magnitude(b);
              IF mag_a = 0 OR mag_b = 0 THEN
                RETURN 0;
              END IF;
              RETURN dp / (mag_a * mag_b);
            END;
            $$ LANGUAGE plpgsql IMMUTABLE;
            """))
            
            db.commit()
            logger.info("Successfully registered custom similarity functions in PostgreSQL.")
        else:
            logger.info("SQLite dialect detected. Skipping PL/pgSQL function registration (vector matching will be calculated in Python).")
            
        # Seed initial data
        seed_data(db)
        
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        db.rollback()
        raise e
    finally:
        db.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    init_database()
