import os
from typing import Any, Dict, List
from tools.base import BaseTool

class FileSearchTool(BaseTool):
    # Static IT Knowledge Base (fallback and mock search repository)
    MOCK_FILES = {
        "network_config.txt": (
            "IT Corporate Network Configuration:\n"
            "- Office Secure Wi-Fi SSID: 'IT-Corp-Secure'\n"
            "- Wi-Fi Security: WPA3-Enterprise (PEAP/MSCHAPv2)\n"
            "- Local VPN Gateway: vpn.internal-corp.net\n"
            "- VPN Protocol: OpenVPN (Port 1194 UDP) or WireGuard (Port 51820 UDP)\n"
            "- Support Email: net-ops@company.com"
        ),
        "printer_setup_instructions.md": (
            "# Main Office Printer Installation Guide\n"
            "To add the corporate high-capacity printer on Windows:\n"
            "1. Open Settings -> Bluetooth & Devices -> Printers & Scanners.\n"
            "2. Click 'Add device', then select 'The printer that I want isn't listed'.\n"
            "3. Select 'Add a printer using an IP address or hostname'.\n"
            "4. Enter Device Type: 'TCP/IP Device' and Hostname/IP: '192.168.4.150'.\n"
            "5. Select Driver: 'HP LaserJet Pro M404-M405' (Standard PCL6 driver).\n"
            "6. Print a test page. If blank, check your VLAN assignment."
        ),
        "email_settings.json": (
            "{\n"
            "  \"service\": \"Microsoft 365 Exchange Online\",\n"
            "  \"imap_server\": \"outlook.office365.com\",\n"
            "  \"imap_port\": 993,\n"
            "  \"imap_encryption\": \"SSL/TLS\",\n"
            "  \"smtp_server\": \"smtp.office365.com\",\n"
            "  \"smtp_port\": 587,\n"
            "  \"smtp_encryption\": \"STARTTLS\",\n"
            "  \"mfa_required\": true\n"
            "}"
        ),
        "active_users_log.txt": (
            "System Audit Report - Active Users:\n"
            "- Total system accounts: 4,512\n"
            "- Currently logged-in active users: 1,847\n"
            "- Service accounts: 45\n"
            "- Locked/suspended accounts: 12\n"
            "- Database cluster node count: 3 (all active and synchronized)"
        ),
        "server_deploy_pipeline.md": (
            "# Deployment Pipeline for Backend Services\n"
            "To deploy the backend IT service:\n"
            "1. Git branch must be merged into 'main'.\n"
            "2. CI pipeline compiles the Docker container and runs pytest.\n"
            "3. Deploy container via Kubernetes: 'kubectl rollout restart deployment/backend-service -n it-helpdesk'.\n"
            "4. Verify database migrations: 'alembic upgrade head' is executed automatically during pod initialization."
        )
    }

    @property
    def name(self) -> str:
        return "file_search"

    @property
    def description(self) -> str:
        return (
            "Search local files or corporate IT knowledge repositories for a specific keyword or query. "
            "Returns a list of matching files with matching content snippets."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search term or keyword to look for (e.g. 'vpn', 'SSID', 'deploy')"
                }
            },
            "required": ["query"]
        }

    def _execute(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        if not query:
            raise ValueError("No search query provided.")
            
        query_lower = query.lower()
        results = []
        
        # 1. Search actual workspace docs/ folder if it exists
        docs_dir = os.path.join(os.getcwd(), "docs")
        if os.path.exists(docs_dir) and os.path.isdir(docs_dir):
            for filename in os.listdir(docs_dir):
                file_path = os.path.join(docs_dir, filename)
                if os.path.isfile(file_path):
                    try:
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read()
                            if query_lower in content.lower():
                                # Extract matching lines
                                matching_lines = []
                                for line in content.splitlines():
                                    if query_lower in line.lower():
                                        matching_lines.append(line.strip())
                                results.append({
                                    "source": "local_disk",
                                    "filename": filename,
                                    "matching_lines": matching_lines[:5], # Cap at 5 lines
                                    "full_content": content if len(content) < 1000 else content[:1000] + "\n[Truncated]"
                                })
                    except Exception as e:
                        # Log error internally but proceed with other files
                        pass
                        
        # 2. Search high-fidelity static mock files
        for filename, content in self.MOCK_FILES.items():
            if query_lower in content.lower() or query_lower in filename.lower():
                matching_lines = []
                for line in content.splitlines():
                    if query_lower in line.lower():
                        matching_lines.append(line.strip())
                
                # Check if this file was already matched physically (to avoid duplicates)
                already_matched = any(r["filename"] == filename for r in results)
                if not already_matched:
                    results.append({
                        "source": "knowledge_repo",
                        "filename": filename,
                        "matching_lines": matching_lines[:5],
                        "full_content": content
                    })
                    
        return results
