#!/usr/bin/env python3
"""
Script zur Analyse der PyCharm-Inspektionsdatei für broad exception catching Probleme.
Gruppiert die Probleme nach Dateien und Modulen für eine strukturierte Task-Erstellung.
"""

import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
import re

def parse_inspection_file(file_path):
    """Parst die PyCharm-Inspektionsdatei und extrahiert alle Probleme."""
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    problems = []
    for problem in root.findall('problem'):
        file_elem = problem.find('file')
        line_elem = problem.find('line')
        description_elem = problem.find('description')
        highlighted_elem = problem.find('highlighted_element')
        
        if file_elem is not None and line_elem is not None:
            file_path = file_elem.text
            # Extrahiere den relativen Pfad
            if '$PROJECT_DIR$/' in file_path:
                file_path = file_path.replace('file://$PROJECT_DIR$/', '')
            
            problems.append({
                'file': file_path,
                'line': int(line_elem.text),
                'description': description_elem.text if description_elem is not None else '',
                'highlighted_element': highlighted_elem.text if highlighted_elem is not None else ''
            })
    
    return problems

def group_problems_by_module(problems):
    """Gruppiert Probleme nach Modulen und Dateien."""
    grouped = defaultdict(lambda: defaultdict(list))
    
    for problem in problems:
        file_path = problem['file']
        
        # Bestimme das Hauptmodul
        path_parts = file_path.split('/')
        if len(path_parts) >= 2:
            main_module = f"{path_parts[0]}/{path_parts[1]}"
        else:
            main_module = path_parts[0] if path_parts else 'unknown'
        
        grouped[main_module][file_path].append(problem)
    
    return grouped

def analyze_problem_severity(file_path, problems_count):
    """Bestimmt die Priorität basierend auf Dateipfad und Anzahl der Probleme."""
    # Hohe Priorität für Security, Core-Funktionalität
    high_priority_patterns = [
        'security/', 'authentication', 'encryption', 'secure_communication',
        'orchestrator/', 'operations/', 'factory/', 'memory/',
        'grpc/', 'middleware/', 'routes/'
    ]
    
    # Mittlere Priorität
    medium_priority_patterns = [
        'agents/', 'workflows/', 'chains/', 'tools/', 'capabilities/'
    ]
    
    # Niedrige Priorität
    low_priority_patterns = [
        'test', 'example', 'demo', 'utils'
    ]
    
    file_lower = file_path.lower()
    
    for pattern in high_priority_patterns:
        if pattern in file_lower:
            return 'HIGH'
    
    for pattern in medium_priority_patterns:
        if pattern in file_lower:
            return 'MEDIUM'
    
    for pattern in low_priority_patterns:
        if pattern in file_lower:
            return 'LOW'
    
    # Fallback basierend auf Anzahl der Probleme
    if problems_count >= 5:
        return 'HIGH'
    elif problems_count >= 3:
        return 'MEDIUM'
    else:
        return 'LOW'

def generate_task_recommendations(grouped_problems):
    """Generiert Task-Empfehlungen basierend auf den gruppierten Problemen."""
    tasks = []
    
    for module, files in grouped_problems.items():
        total_problems = sum(len(problems) for problems in files.values())
        
        # Gruppiere Dateien nach Priorität
        high_priority_files = []
        medium_priority_files = []
        low_priority_files = []
        
        for file_path, problems in files.items():
            priority = analyze_problem_severity(file_path, len(problems))
            file_info = {
                'path': file_path,
                'problems': problems,
                'count': len(problems),
                'priority': priority
            }
            
            if priority == 'HIGH':
                high_priority_files.append(file_info)
            elif priority == 'MEDIUM':
                medium_priority_files.append(file_info)
            else:
                low_priority_files.append(file_info)
        
        # Erstelle Tasks für jede Prioritätsstufe
        for priority, files_list in [('HIGH', high_priority_files), 
                                   ('MEDIUM', medium_priority_files), 
                                   ('LOW', low_priority_files)]:
            if files_list:
                tasks.append({
                    'module': module,
                    'priority': priority,
                    'files': files_list,
                    'total_problems': sum(f['count'] for f in files_list)
                })
    
    return tasks

def print_analysis_report(tasks):
    """Druckt einen detaillierten Analysebericht."""
    print("=" * 80)
    print("ANALYSE DER BROAD EXCEPTION CATCHING PROBLEME")
    print("=" * 80)
    
    total_problems = sum(task['total_problems'] for task in tasks)
    print(f"\nGESAMTÜBERSICHT:")
    print(f"- Gesamtanzahl Probleme: {total_problems}")
    print(f"- Anzahl Module: {len(set(task['module'] for task in tasks))}")
    print(f"- Anzahl Tasks: {len(tasks)}")
    
    # Sortiere Tasks nach Priorität
    priority_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
    tasks.sort(key=lambda x: (priority_order[x['priority']], -x['total_problems']))
    
    print(f"\nDETAILLIERTE AUFSCHLÜSSELUNG:")
    print("-" * 80)
    
    for i, task in enumerate(tasks, 1):
        print(f"\nTask {i}: {task['module']} - Priorität: {task['priority']}")
        print(f"Probleme: {task['total_problems']}")
        print(f"Dateien ({len(task['files'])}):")
        
        for file_info in sorted(task['files'], key=lambda x: -x['count']):
            print(f"  - {file_info['path']} ({file_info['count']} Probleme)")
            # Zeige erste paar Zeilennummern als Beispiel
            lines = [str(p['line']) for p in file_info['problems'][:5]]
            if len(file_info['problems']) > 5:
                lines.append(f"... +{len(file_info['problems']) - 5} weitere")
            print(f"    Zeilen: {', '.join(lines)}")

if __name__ == "__main__":
    # Analysiere die Inspektionsdatei
    inspection_file = "code-report/edited/PyBroadExceptionInspection.xml"
    
    print("Analysiere PyCharm-Inspektionsdatei...")
    problems = parse_inspection_file(inspection_file)
    print(f"Gefunden: {len(problems)} Probleme")
    
    print("Gruppiere Probleme nach Modulen...")
    grouped = group_problems_by_module(problems)
    
    print("Generiere Task-Empfehlungen...")
    tasks = generate_task_recommendations(grouped)
    
    print_analysis_report(tasks)
