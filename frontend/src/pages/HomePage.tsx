import { ArrowRight, BookOpen, Compass, MessageCircle, Rocket, Server, StickyNote } from 'lucide-react';
import React from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';

const askLibrary = {
  title: 'Ask the library',
  description: 'Ask a question and follow the sources cited in the answer.',
  to: '/chat',
  icon: MessageCircle,
};

const researchTasks = [
  askLibrary,
  {
    title: 'Run an investigation',
    description: 'Set a research goal and a run budget in a campaign.',
    to: '/campaigns',
    icon: Rocket,
  },
  {
    title: 'Monitor a topic',
    description: 'Run recurring searches and collect new evidence on a schedule.',
    to: '/settings/domain-profiles',
    icon: Compass,
  },
  {
    title: 'Share context',
    description: 'Give an AI client access to selected sources, notes, or research tools through MCP.',
    to: '/mcp-config',
    icon: Server,
  },
];

const viewerTasks = [
  askLibrary,
  {
    title: 'Browse documents',
    description: 'Explore the sources in the library.',
    to: '/documents',
    icon: BookOpen,
  },
  {
    title: 'Read research notes',
    description: 'Explore findings and follow their evidence.',
    to: '/research-notes',
    icon: StickyNote,
  },
];

const HomePage: React.FC = () => {
  const { user } = useAuth();
  const tasks = user?.role === 'viewer' ? viewerTasks : researchTasks;

  return (
    <section aria-labelledby="home-heading" className="w-full max-w-4xl px-4 py-6 sm:px-6 sm:py-8">
      <h2 id="home-heading" className="mb-5 text-2xl font-semibold tracking-tight text-gray-900">
        What would you like to do?
      </h2>
      <ul className="grid gap-3 sm:grid-cols-2">
        {tasks.map(({ title, description, to, icon: Icon }) => (
          <li key={to}>
            <Link
              to={to}
              className="group flex h-full items-start gap-3 rounded-lg border border-gray-200 bg-white p-4 hover:border-primary-400 focus-visible:outline focus-visible:outline-2 focus-visible:outline-primary-600"
            >
              <Icon aria-hidden="true" className="mt-0.5 h-5 w-5 shrink-0 text-primary-700" />
              <div className="min-w-0 flex-1">
                <h3 className="font-medium text-gray-900">{title}</h3>
                <p className="mt-1 text-sm text-gray-600">{description}</p>
              </div>
              <ArrowRight aria-hidden="true" className="mt-0.5 h-4 w-4 shrink-0 text-gray-500 group-hover:text-primary-700" />
            </Link>
          </li>
        ))}
      </ul>
    </section>
  );
};

export default HomePage;
