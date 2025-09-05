#!/usr/bin/env python3
"""
Enterprise Dashboard Data Collector

Collects real-time data from GitHub Actions workflows for enterprise dashboard
generation and analytics.
"""

import json
import os
import sys
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any
import argparse


class GitHubActionsDataCollector:
    """Collects comprehensive data from GitHub Actions for enterprise dashboard."""
    
    def __init__(self, github_token: str, repo_owner: str, repo_name: str):
        """Initialize the data collector.
        
        Args:
            github_token: GitHub API token
            repo_owner: Repository owner
            repo_name: Repository name
        """
        self.github_token = github_token
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.base_url = "https://api.github.com"
        self.headers = {
            "Authorization": f"token {github_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
    def get_workflow_runs(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get workflow runs for the specified period.
        
        Args:
            days: Number of days to look back
            
        Returns:
            List of workflow run data
        """
        since = (datetime.utcnow() - timedelta(days=days)).isoformat()
        
        url = f"{self.base_url}/repos/{self.repo_owner}/{self.repo_name}/actions/runs"
        params = {
            "per_page": 100,
            "created": f">={since}"
        }
        
        all_runs = []
        page = 1
        
        while True:
            params["page"] = page
            response = requests.get(url, headers=self.headers, params=params)
            
            if response.status_code != 200:
                print(f"Error fetching workflow runs: {response.status_code}")
                break
                
            data = response.json()
            runs = data.get("workflow_runs", [])
            
            if not runs:
                break
                
            all_runs.extend(runs)
            
            # Check if we've reached the date limit
            last_run_date = datetime.fromisoformat(runs[-1]["created_at"].replace("Z", "+00:00"))
            if last_run_date < datetime.utcnow().replace(tzinfo=last_run_date.tzinfo) - timedelta(days=days):
                break
                
            page += 1
            
            # Limit to prevent excessive API calls
            if page > 10:
                break
        
        return all_runs
    
    def get_workflows(self) -> List[Dict[str, Any]]:
        """Get all workflows in the repository.
        
        Returns:
            List of workflow data
        """
        url = f"{self.base_url}/repos/{self.repo_owner}/{self.repo_name}/actions/workflows"
        response = requests.get(url, headers=self.headers)
        
        if response.status_code != 200:
            print(f"Error fetching workflows: {response.status_code}")
            return []
            
        return response.json().get("workflows", [])
    
    def analyze_workflow_performance(self, runs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze workflow performance metrics.
        
        Args:
            runs: List of workflow runs
            
        Returns:
            Performance analysis data
        """
        if not runs:
            return {}
        
        # Group runs by workflow
        workflow_stats = {}
        total_runs = len(runs)
        successful_runs = 0
        total_duration = 0
        
        for run in runs:
            workflow_name = run.get("name", "Unknown")
            status = run.get("conclusion", "unknown")
            
            if workflow_name not in workflow_stats:
                workflow_stats[workflow_name] = {
                    "total_runs": 0,
                    "successful_runs": 0,
                    "failed_runs": 0,
                    "total_duration": 0,
                    "durations": []
                }
            
            workflow_stats[workflow_name]["total_runs"] += 1
            
            if status == "success":
                workflow_stats[workflow_name]["successful_runs"] += 1
                successful_runs += 1
            elif status in ["failure", "cancelled", "timed_out"]:
                workflow_stats[workflow_name]["failed_runs"] += 1
            
            # Calculate duration
            if run.get("created_at") and run.get("updated_at"):
                created = datetime.fromisoformat(run["created_at"].replace("Z", "+00:00"))
                updated = datetime.fromisoformat(run["updated_at"].replace("Z", "+00:00"))
                duration = (updated - created).total_seconds() / 60  # minutes
                
                workflow_stats[workflow_name]["total_duration"] += duration
                workflow_stats[workflow_name]["durations"].append(duration)
                total_duration += duration
        
        # Calculate overall metrics
        overall_success_rate = (successful_runs / total_runs * 100) if total_runs > 0 else 0
        average_duration = total_duration / total_runs if total_runs > 0 else 0
        
        # Calculate per-workflow metrics
        for workflow_name, stats in workflow_stats.items():
            stats["success_rate"] = (stats["successful_runs"] / stats["total_runs"] * 100) if stats["total_runs"] > 0 else 0
            stats["average_duration"] = stats["total_duration"] / stats["total_runs"] if stats["total_runs"] > 0 else 0
            
            # Determine trend (simplified)
            if len(stats["durations"]) >= 5:
                recent_avg = sum(stats["durations"][-5:]) / 5
                older_avg = sum(stats["durations"][:-5]) / len(stats["durations"][:-5]) if len(stats["durations"]) > 5 else recent_avg
                
                if recent_avg < older_avg * 0.9:
                    stats["trend"] = "improving"
                elif recent_avg > older_avg * 1.1:
                    stats["trend"] = "degrading"
                else:
                    stats["trend"] = "stable"
            else:
                stats["trend"] = "stable"
        
        return {
            "total_workflows": len(workflow_stats),
            "total_runs": total_runs,
            "overall_success_rate": round(overall_success_rate, 1),
            "average_duration": round(average_duration, 1),
            "workflow_details": workflow_stats
        }
    
    def estimate_costs(self, runs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Estimate GitHub Actions costs based on usage.
        
        Args:
            runs: List of workflow runs
            
        Returns:
            Cost estimation data
        """
        # GitHub Actions pricing (approximate)
        COST_PER_MINUTE = {
            "ubuntu": 0.008,  # $0.008 per minute
            "windows": 0.016,  # $0.016 per minute
            "macos": 0.08     # $0.08 per minute
        }
        
        total_cost = 0
        runner_minutes = {"ubuntu": 0, "windows": 0, "macos": 0}
        
        for run in runs:
            # Estimate runner type based on workflow name (simplified)
            runner_type = "ubuntu"  # Default
            if "windows" in run.get("name", "").lower():
                runner_type = "windows"
            elif "macos" in run.get("name", "").lower():
                runner_type = "macos"
            
            # Calculate duration
            if run.get("created_at") and run.get("updated_at"):
                created = datetime.fromisoformat(run["created_at"].replace("Z", "+00:00"))
                updated = datetime.fromisoformat(run["updated_at"].replace("Z", "+00:00"))
                duration_minutes = (updated - created).total_seconds() / 60
                
                runner_minutes[runner_type] += duration_minutes
                total_cost += duration_minutes * COST_PER_MINUTE[runner_type]
        
        return {
            "total_cost_usd": round(total_cost, 2),
            "runner_minutes": {
                "ubuntu": round(runner_minutes["ubuntu"], 1),
                "windows": round(runner_minutes["windows"], 1),
                "macos": round(runner_minutes["macos"], 1)
            },
            "cost_breakdown": {
                "ubuntu": round(runner_minutes["ubuntu"] * COST_PER_MINUTE["ubuntu"], 2),
                "windows": round(runner_minutes["windows"] * COST_PER_MINUTE["windows"], 2),
                "macos": round(runner_minutes["macos"] * COST_PER_MINUTE["macos"], 2)
            }
        }
    
    def collect_comprehensive_data(self, days: int = 30) -> Dict[str, Any]:
        """Collect comprehensive dashboard data.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Comprehensive dashboard data
        """
        print(f"🔍 Collecting GitHub Actions data for last {days} days...")
        
        # Collect raw data
        workflows = self.get_workflows()
        runs = self.get_workflow_runs(days)
        
        print(f"📊 Found {len(workflows)} workflows and {len(runs)} runs")
        
        # Analyze data
        performance_data = self.analyze_workflow_performance(runs)
        cost_data = self.estimate_costs(runs)
        
        # Generate comprehensive report
        dashboard_data = {
            "generated_at": datetime.utcnow().isoformat(),
            "period_days": days,
            "repository": f"{self.repo_owner}/{self.repo_name}",
            "summary": {
                "total_workflows": len(workflows),
                "total_runs": len(runs),
                "success_rate": performance_data.get("overall_success_rate", 0),
                "average_duration": performance_data.get("average_duration", 0),
                "estimated_cost": cost_data.get("total_cost_usd", 0)
            },
            "performance_metrics": performance_data,
            "cost_analysis": cost_data,
            "workflows": workflows[:10],  # Limit to first 10 workflows
            "recent_runs": runs[:20]  # Limit to 20 most recent runs
        }
        
        return dashboard_data
    
    def save_data(self, data: Dict[str, Any], output_file: str = "dashboard-data.json"):
        """Save collected data to file.
        
        Args:
            data: Dashboard data to save
            output_file: Output file path
        """
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"✅ Dashboard data saved to {output_file}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Collect GitHub Actions data for enterprise dashboard")
    parser.add_argument("--token", required=True, help="GitHub API token")
    parser.add_argument("--owner", required=True, help="Repository owner")
    parser.add_argument("--repo", required=True, help="Repository name")
    parser.add_argument("--days", type=int, default=30, help="Number of days to analyze")
    parser.add_argument("--output", default="dashboard-data.json", help="Output file path")
    
    args = parser.parse_args()
    
    # Initialize collector
    collector = GitHubActionsDataCollector(args.token, args.owner, args.repo)
    
    # Collect data
    try:
        data = collector.collect_comprehensive_data(args.days)
        collector.save_data(data, args.output)
        
        print("✅ Data collection completed successfully!")
        print("📊 Summary:")
        print(f"  - Workflows: {data['summary']['total_workflows']}")
        print(f"  - Runs: {data['summary']['total_runs']}")
        print(f"  - Success Rate: {data['summary']['success_rate']}%")
        print(f"  - Average Duration: {data['summary']['average_duration']} minutes")
        print(f"  - Estimated Cost: ${data['summary']['estimated_cost']}")
        
    except Exception as e:
        print(f"❌ Error collecting data: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
