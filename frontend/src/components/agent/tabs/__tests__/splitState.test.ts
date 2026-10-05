import fs from 'fs';
import path from 'path';

/**
 * Extracting a tab out of AutonomousAgentsPage moves state across a component
 * boundary, and there is one way to get that wrong that nothing else catches:
 * leave the state declared on BOTH sides. The page's mutation then writes one
 * `useState` while the tab reads a different one that happens to share its
 * name. It type-checks, it renders, every test passes -- edits just silently
 * fail to appear.
 *
 * That shipped once during the health-tab extraction (`healthPolicyDrafts`,
 * written by a page mutation and by the tab). The rule it violated:
 * **state lives where its writers are, not where its readers are.** If the page
 * still writes it, it stays on the page and goes down as a prop.
 */
const PAGE = path.join(__dirname, '../../../../pages/AutonomousAgentsPage.tsx');
const TABS_DIR = path.join(__dirname, '..');

const allMatches = (src: string, re: RegExp): string[] => {
  const out: string[] = [];
  let m = re.exec(src);
  while (m !== null) {
    out.push(m[1]);
    m = re.exec(src);
  }
  return out;
};

const declaredStates = (src: string): string[] =>
  allMatches(src, /const \[(\w+), set\w+\] = useState/g);

const calledSetters = (src: string): Set<string> =>
  new Set(allMatches(src, /\b(set[A-Z]\w*)\s*\(/g));

const setterFor = (state: string) => `set${state[0].toUpperCase()}${state.slice(1)}`;

describe('extracted agent tabs do not split state with the page', () => {
  const pageSrc = fs.readFileSync(PAGE, 'utf8');
  const pageDeclares = new Set(declaredStates(pageSrc));
  const pageWrites = calledSetters(pageSrc);
  const tabs = fs.readdirSync(TABS_DIR).filter((f) => f.endsWith('.tsx'));

  it('finds the extracted tabs', () => {
    expect(tabs.length).toBeGreaterThan(0);
  });

  it.each(tabs)('%s declares no state the page also declares', (tab) => {
    const tabDeclares = declaredStates(fs.readFileSync(path.join(TABS_DIR, tab), 'utf8'));
    expect(tabDeclares.filter((s) => pageDeclares.has(s))).toEqual([]);
  });

  it.each(tabs)('%s owns no state the page still writes', (tab) => {
    const tabDeclares = declaredStates(fs.readFileSync(path.join(TABS_DIR, tab), 'utf8'));
    expect(tabDeclares.filter((s) => pageWrites.has(setterFor(s)))).toEqual([]);
  });
});
