/**
 * The three coding-swarm quick starts, as data.
 *
 * They were three components declared inside AutonomousAgentsPage's render
 * body, and each was the same JSX: `<QuickStartCodingSwarmModal>` with a
 * different title, colour, default goal and mutation. Sixty-six lines of
 * near-duplicate markup, three more component identities churning on every
 * render, and a fourth preset meaning a fourth copy.
 *
 * What varies between them is text, a colour and which mutation to call —
 * which is a table. What does not vary is the modal, which stays one
 * component.
 */

export interface SwarmQuickStartPreset {
  /** Matches the `presetKey` the launch seed carries, so a seeded launch
   *  reaches the right modal. */
  presetKey: 'bug_triage_swarm' | 'build_break_swarm' | 'frontend_regression_swarm';
  title: string;
  description: string;
  /** Prefix for the default run name; the date is appended at render time. */
  namePrefix: string;
  /** Used when the operator has not typed a goal hint of their own. */
  failureSymptomPlaceholder: string;
  defaultGoal: string;
  defaultScope: 'auto' | 'backend' | 'frontend' | 'worker';
  accentClassName: string;
  submitLabel: string;
}

export const SWARM_QUICK_START_PRESETS: SwarmQuickStartPreset[] = [
  {
    presetKey: 'bug_triage_swarm',
    title: 'Quick Start Bug Triage Swarm',
    description:
      'Launch a coding swarm with reproducer, root-cause, patcher, and verifier slices. High-confidence fan-in auto-launches the existing repair loop.',
    namePrefix: 'Bug Triage Swarm',
    failureSymptomPlaceholder: 'Describe the observed bug or failing behavior',
    defaultGoal:
      'Reproduce the failure, rank the best repair path, and auto-launch the repair loop when confidence is high',
    defaultScope: 'auto',
    accentClassName: 'border-rose-100 bg-rose-50 text-rose-800',
    submitLabel: 'Start Swarm',
  },
  {
    presetKey: 'build_break_swarm',
    title: 'Quick Start Build Break Swarm',
    description:
      'Launch a coding swarm tuned for compiler failures, broken imports, and build-step regressions. High-confidence fan-in still hands off into the repair chain.',
    namePrefix: 'Build Break Swarm',
    failureSymptomPlaceholder: 'Describe the failing build, compile, or test command',
    defaultGoal:
      'Identify the broken build step, isolate the minimal file cluster, and auto-launch repair when the swarm converges',
    defaultScope: 'backend',
    accentClassName: 'border-amber-100 bg-amber-50 text-amber-800',
    submitLabel: 'Start Build Swarm',
  },
  {
    presetKey: 'frontend_regression_swarm',
    title: 'Quick Start Frontend Regression Swarm',
    description:
      'Launch a coding swarm tuned for UI regressions, state mismatches, and page-level breakage, while keeping the same fan-in and repair-chain handoff model.',
    namePrefix: 'Frontend Regression Swarm',
    failureSymptomPlaceholder:
      'Describe the broken page, interaction, or visible UI regression',
    defaultGoal:
      'Reproduce the frontend regression, narrow the affected page/component cluster, and promote the winning path into repair when safe',
    defaultScope: 'frontend',
    accentClassName: 'border-cyan-100 bg-cyan-50 text-cyan-800',
    submitLabel: 'Start Frontend Swarm',
  },
];

export const swarmQuickStartPreset = (
  presetKey: string
): SwarmQuickStartPreset | undefined =>
  SWARM_QUICK_START_PRESETS.find((preset) => preset.presetKey === presetKey);
