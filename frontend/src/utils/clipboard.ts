/**
 * Copying a value to the clipboard, with the toast the user expects either
 * way.
 *
 * Module-scope in AutonomousAgentsPage until components started moving out of
 * it and needed the same behaviour. Nothing here touches component state.
 */

import toast from 'react-hot-toast';

/** Copy `text`, telling the user what happened. `label` names the thing. */
export const copyText = async (text: string, label: string): Promise<void> => {
  if (!text) {
    toast.error(`Nothing to copy for ${label}`);
    return;
  }
  try {
    if (navigator?.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
      toast.success(`${label} copied`);
      return;
    }
    toast.error('Clipboard not supported');
  } catch {
    toast.error(`Failed to copy ${label}`);
  }
};
