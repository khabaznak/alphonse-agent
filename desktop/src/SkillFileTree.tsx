import { useEffect, useMemo, useState } from "react";
import { ChevronDown, ChevronRight, FileText, Folder, FolderOpen } from "lucide-react";

export type SkillFile = { path: string; size_bytes: number; text_editable: boolean };
type Node = { name: string; path: string; children: Node[]; file?: SkillFile };

export function buildSkillFileTree(files: SkillFile[], directories: string[] = []): Node[] {
  const root: Node[] = [];
  for (const entry of [...directories.map(path => ({ path, file: undefined })), ...files.map(file => ({ path: file.path, file }))]) {
    const parts = entry.path.split("/").filter(Boolean);
    let children = root;
    for (let index = 0; index < parts.length; index++) {
      const name = parts[index];
      const path = parts.slice(0, index + 1).join("/");
      let node = children.find(item => item.name === name);
      if (!node) { node = { name, path, children: [] }; children.push(node); }
      if (index === parts.length - 1 && entry.file) node.file = entry.file;
      children = node.children;
    }
  }
  const sort = (nodes: Node[]) => { nodes.sort((a, b) => Number(Boolean(a.file)) - Number(Boolean(b.file)) || a.name.localeCompare(b.name)); nodes.forEach(node => sort(node.children)); };
  sort(root);
  return root;
}

export function SkillFileTree({ files, directories, selectedPath, onSelect }: { files: SkillFile[]; directories: string[]; selectedPath: string; onSelect: (path: string) => void }) {
  const tree = useMemo(() => buildSkillFileTree(files, directories), [files, directories]);
  const [expanded, setExpanded] = useState<Set<string>>(() => new Set());
  useEffect(() => {
    const parts = selectedPath.split("/");
    setExpanded(current => { const next = new Set(current); for (let i = 1; i < parts.length; i++) next.add(parts.slice(0, i).join("/")); return next; });
  }, [selectedPath]);
  const renderNodes = (nodes: Node[]) => <ul>{nodes.map(node => {
    const open = expanded.has(node.path);
    if (node.file) return <li key={node.path}><button type="button" className={`skill-tree-item skill-tree-file${selectedPath === node.path ? " active" : ""}`} aria-current={selectedPath === node.path ? "true" : undefined} onClick={() => onSelect(node.path)} title={node.path}><FileText size={15} aria-hidden="true" /><span>{node.name}</span>{!node.file.text_editable && <small>Binary or large</small>}</button></li>;
    return <li key={node.path}><button type="button" className="skill-tree-item skill-tree-folder" aria-expanded={open} onClick={() => setExpanded(current => { const next = new Set(current); if (next.has(node.path)) next.delete(node.path); else next.add(node.path); return next; })}>{open ? <ChevronDown size={14} aria-hidden="true" /> : <ChevronRight size={14} aria-hidden="true" />}{open ? <FolderOpen size={15} aria-hidden="true" /> : <Folder size={15} aria-hidden="true" />}<span>{node.name}</span></button>{open && renderNodes(node.children)}</li>;
  })}</ul>;
  return <nav className="skill-file-tree" aria-label="Skill package files">{tree.length ? renderNodes(tree) : <p>No package files available.</p>}</nav>;
}
