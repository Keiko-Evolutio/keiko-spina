#!/usr/bin/env python3
"""
Metrics Analysis Script for GitHub Actions Monitoring

This script analyzes collected metrics from GitHub Actions workflows
and generates insights, trends, and recommendations.

Usage:
    python scripts/analyze-metrics.py [--input-dir DIR] [--output-dir DIR] [--days N]
"""

import argparse
import csv
import json
# os und sys werden nicht verwendet - entfernt
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import statistics


class MetricsAnalyzer:
    """Analyzes GitHub Actions monitoring metrics."""
    
    def __init__(self, input_dir: str = "monitoring-data", output_dir: str = "analytics-reports"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.metrics = []
        self.analysis_results = {}
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
    
    def load_metrics(self, days: int = 7) -> None:
        """Load metrics from CSV files within the specified time range."""
        print(f"📊 Loading metrics from {self.input_dir} (last {days} days)")
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Find all CSV files
        csv_files = list(self.input_dir.glob("**/*.csv"))
        
        if not csv_files:
            print("⚠️ No CSV files found in input directory")
            return
        
        for csv_file in csv_files:
            try:
                with open(csv_file, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Skip header row if it appears in data
                        if row.get('metric_name') == 'metric_name':
                            continue
                        
                        # Parse timestamp
                        try:
                            timestamp = datetime.strptime(row['timestamp'], '%Y-%m-%d %H:%M:%S UTC')
                            if timestamp >= cutoff_date:
                                self.metrics.append(row)
                        except (ValueError, KeyError):
                            continue
                            
            except Exception as e:
                print(f"⚠️ Error reading {csv_file}: {e}")
        
        print(f"✅ Loaded {len(self.metrics)} metrics from {len(csv_files)} files")
    
    def analyze_workflow_performance(self) -> Dict[str, Any]:
        """Analyze workflow performance metrics."""
        print("📈 Analyzing workflow performance...")
        
        workflow_metrics = defaultdict(list)
        job_metrics = defaultdict(list)
        
        for metric in self.metrics:
            metric_name = metric.get('metric_name', '')
            tags = metric.get('tags', '')
            value = metric.get('value', '0')
            
            try:
                numeric_value = float(value)
            except ValueError:
                continue
            
            # Extract workflow and job from tags
            workflow = self._extract_tag_value(tags, 'workflow')
            job = self._extract_tag_value(tags, 'job')
            
            if 'workflow.duration' in metric_name and workflow:
                workflow_metrics[workflow].append(numeric_value)
            elif 'job.execution' in metric_name and job:
                job_metrics[job].append(numeric_value)
        
        # Calculate statistics
        workflow_stats = {}
        for workflow, durations in workflow_metrics.items():
            if durations:
                workflow_stats[workflow] = {
                    'avg_duration': statistics.mean(durations),
                    'median_duration': statistics.median(durations),
                    'min_duration': min(durations),
                    'max_duration': max(durations),
                    'total_runs': len(durations)
                }
        
        job_stats = {}
        for job, executions in job_metrics.items():
            if executions:
                job_stats[job] = {
                    'total_executions': len(executions),
                    'avg_executions': statistics.mean(executions)
                }
        
        return {
            'workflow_performance': workflow_stats,
            'job_performance': job_stats,
            'total_workflows': len(workflow_stats),
            'total_jobs': len(job_stats)
        }
    
    def analyze_failure_patterns(self) -> Dict[str, Any]:
        """Analyze failure patterns and error categories."""
        print("🔍 Analyzing failure patterns...")
        
        failures = []
        error_categories = Counter()
        failure_trends = defaultdict(int)
        
        for metric in self.metrics:
            metric_name = metric.get('metric_name', '')
            tags = metric.get('tags', '')
            timestamp = metric.get('timestamp', '')
            
            if 'failure' in metric_name or 'error' in metric_name:
                failures.append(metric)
                
                # Extract error category
                category = self._extract_tag_value(tags, 'category')
                if category:
                    error_categories[category] += 1
                
                # Track daily failure trends
                try:
                    date = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S UTC').date()
                    failure_trends[str(date)] += 1
                except ValueError:
                    continue
        
        return {
            'total_failures': len(failures),
            'error_categories': dict(error_categories),
            'failure_trends': dict(failure_trends),
            'most_common_errors': error_categories.most_common(5)
        }
    
    def analyze_system_resources(self) -> Dict[str, Any]:
        """Analyze system resource utilization."""
        print("🖥️ Analyzing system resources...")
        
        cpu_usage = []
        memory_usage = []
        disk_usage = []
        
        for metric in self.metrics:
            metric_name = metric.get('metric_name', '')
            value = metric.get('value', '0')
            
            try:
                numeric_value = float(value)
            except ValueError:
                continue
            
            if 'cpu_usage' in metric_name:
                cpu_usage.append(numeric_value)
            elif 'memory_usage_percent' in metric_name:
                memory_usage.append(numeric_value)
            elif 'disk_usage_percent' in metric_name:
                disk_usage.append(numeric_value)
        
        def calculate_stats(values: List[float]) -> Dict[str, float]:
            if not values:
                return {}
            return {
                'avg': statistics.mean(values),
                'median': statistics.median(values),
                'min': min(values),
                'max': max(values),
                'p95': sorted(values)[int(len(values) * 0.95)] if len(values) > 20 else max(values)
            }
        
        return {
            'cpu_stats': calculate_stats(cpu_usage),
            'memory_stats': calculate_stats(memory_usage),
            'disk_stats': calculate_stats(disk_usage),
            'resource_samples': {
                'cpu': len(cpu_usage),
                'memory': len(memory_usage),
                'disk': len(disk_usage)
            }
        }
    
    def analyze_test_metrics(self) -> Dict[str, Any]:
        """Analyze test execution metrics."""
        print("🧪 Analyzing test metrics...")
        
        test_types = Counter()
        test_results = defaultdict(list)
        
        for metric in self.metrics:
            metric_name = metric.get('metric_name', '')
            tags = metric.get('tags', '')
            value = metric.get('value', '0')
            
            if 'test' in metric_name:
                test_type = self._extract_tag_value(tags, 'test_type')
                result = self._extract_tag_value(tags, 'result')
                
                if test_type:
                    test_types[test_type] += 1
                    
                    try:
                        numeric_value = float(value)
                        test_results[test_type].append({
                            'value': numeric_value,
                            'result': result,
                            'timestamp': metric.get('timestamp', '')
                        })
                    except ValueError:
                        continue
        
        # Calculate success rates
        success_rates = {}
        for test_type, results in test_results.items():
            if results:
                successful = sum(1 for r in results if r.get('result') == 'success')
                total = len(results)
                success_rates[test_type] = (successful / total) * 100 if total > 0 else 0
        
        return {
            'test_type_distribution': dict(test_types),
            'success_rates': success_rates,
            'total_test_executions': sum(test_types.values())
        }
    
    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on analysis."""
        print("💡 Generating recommendations...")
        
        recommendations = []
        
        # Check workflow performance
        workflow_perf = self.analysis_results.get('workflow_performance', {})
        for workflow, stats in workflow_perf.items():
            avg_duration = stats.get('avg_duration', 0)
            if avg_duration > 1800:  # 30 minutes
                recommendations.append(
                    f"🐌 Workflow '{workflow}' has high average duration ({avg_duration/60:.1f}m). "
                    "Consider optimizing or parallelizing jobs."
                )
        
        # Check failure patterns
        failure_analysis = self.analysis_results.get('failure_analysis', {})
        total_failures = failure_analysis.get('total_failures', 0)
        if total_failures > 10:
            recommendations.append(
                f"⚠️ High failure count ({total_failures}). "
                "Review error patterns and implement better error handling."
            )
        
        # Check resource usage
        resource_analysis = self.analysis_results.get('resource_analysis', {})
        memory_stats = resource_analysis.get('memory_stats', {})
        if memory_stats.get('p95', 0) > 80:
            recommendations.append(
                "🧠 High memory usage detected (>80% at P95). "
                "Consider optimizing memory usage or increasing runner resources."
            )
        
        # Check test success rates
        test_analysis = self.analysis_results.get('test_analysis', {})
        success_rates = test_analysis.get('success_rates', {})
        for test_type, rate in success_rates.items():
            if rate < 95:
                recommendations.append(
                    f"🧪 Low success rate for {test_type} tests ({rate:.1f}%). "
                    "Investigate flaky tests and improve reliability."
                )
        
        if not recommendations:
            recommendations.append("✅ No critical issues detected. System is performing well!")
        
        return recommendations
    
    def _extract_tag_value(self, tags: str, key: str) -> Optional[str]:
        """Extract value for a specific tag key."""
        if not tags:
            return None
        
        for tag in tags.split(','):
            if '=' in tag:
                tag_key, tag_value = tag.split('=', 1)
                if tag_key.strip() == key:
                    return tag_value.strip()
        return None
    
    def run_analysis(self, days: int = 7) -> None:
        """Run complete analysis."""
        print(f"🚀 Starting metrics analysis for last {days} days")
        
        # Load metrics
        self.load_metrics(days)
        
        if not self.metrics:
            print("❌ No metrics found. Exiting.")
            return
        
        # Run analyses
        self.analysis_results['workflow_performance'] = self.analyze_workflow_performance()
        self.analysis_results['failure_analysis'] = self.analyze_failure_patterns()
        self.analysis_results['resource_analysis'] = self.analyze_system_resources()
        self.analysis_results['test_analysis'] = self.analyze_test_metrics()
        
        # Generate recommendations
        recommendations = self.generate_recommendations()
        self.analysis_results['recommendations'] = recommendations
        
        # Save results
        self.save_results()
        
        print("✅ Analysis completed successfully!")
    
    def save_results(self) -> None:
        """Save analysis results to files."""
        print("💾 Saving analysis results...")
        
        # Save JSON report
        json_file = self.output_dir / f"metrics-analysis-{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_file, 'w') as f:
            json.dump(self.analysis_results, f, indent=2, default=str)
        
        # Save markdown report
        md_file = self.output_dir / f"metrics-report-{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(md_file, 'w') as f:
            f.write(self._generate_markdown_report())
        
        print(f"📊 Results saved to {json_file} and {md_file}")
    
    def _generate_markdown_report(self) -> str:
        """Generate markdown report."""
        report = f"""# 📊 Metrics Analysis Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}  
**Total Metrics**: {len(self.metrics)}

## 📈 Workflow Performance

"""
        
        workflow_perf = self.analysis_results.get('workflow_performance', {})
        if workflow_perf:
            report += "| Workflow | Avg Duration | Total Runs | Min | Max |\n"
            report += "|----------|--------------|------------|-----|-----|\n"
            for workflow, stats in workflow_perf.items():
                avg_dur = stats.get('avg_duration', 0) / 60  # Convert to minutes
                total_runs = stats.get('total_runs', 0)
                min_dur = stats.get('min_duration', 0) / 60
                max_dur = stats.get('max_duration', 0) / 60
                report += f"| {workflow} | {avg_dur:.1f}m | {total_runs} | {min_dur:.1f}m | {max_dur:.1f}m |\n"
        
        report += "\n## 🔍 Failure Analysis\n\n"
        
        failure_analysis = self.analysis_results.get('failure_analysis', {})
        total_failures = failure_analysis.get('total_failures', 0)
        error_categories = failure_analysis.get('error_categories', {})
        
        report += f"**Total Failures**: {total_failures}\n\n"
        
        if error_categories:
            report += "### Error Categories\n\n"
            for category, count in error_categories.items():
                report += f"- **{category}**: {count}\n"
        
        report += "\n## 🖥️ Resource Analysis\n\n"
        
        resource_analysis = self.analysis_results.get('resource_analysis', {})
        cpu_stats = resource_analysis.get('cpu_stats', {})
        memory_stats = resource_analysis.get('memory_stats', {})
        
        if cpu_stats:
            report += f"**CPU Usage**: Avg {cpu_stats.get('avg', 0):.1f}%, P95 {cpu_stats.get('p95', 0):.1f}%\n"
        if memory_stats:
            report += f"**Memory Usage**: Avg {memory_stats.get('avg', 0):.1f}%, P95 {memory_stats.get('p95', 0):.1f}%\n"
        
        report += "\n## 💡 Recommendations\n\n"
        
        recommendations = self.analysis_results.get('recommendations', [])
        for rec in recommendations:
            report += f"- {rec}\n"
        
        return report


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Analyze GitHub Actions monitoring metrics")
    parser.add_argument('--input-dir', default='monitoring-data', help='Input directory with metrics')
    parser.add_argument('--output-dir', default='analytics-reports', help='Output directory for reports')
    parser.add_argument('--days', type=int, default=7, help='Number of days to analyze')
    
    args = parser.parse_args()
    
    analyzer = MetricsAnalyzer(args.input_dir, args.output_dir)
    analyzer.run_analysis(args.days)


if __name__ == '__main__':
    main()
