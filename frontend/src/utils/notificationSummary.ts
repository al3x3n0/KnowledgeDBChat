import { AutonomyBudgetNotificationData, CustomerAutonomyBudgetNotificationData, ExperimentRunNotificationData, FollowUpOutcomeNotificationData, HypothesisReevaluationNotificationData, Notification, QueueUrgencyNotificationData } from '../types';

export interface ExperimentNotificationSummary {
  badges: string[];
  reason: string;
  nextAction: string;
  latestOperator: string;
  latestOperatorNote: string;
  latestOperatorOutcome: string;
  latestOperatorOutcomeReason: string;
}

export interface NotificationToastSummary {
  title: string;
  description: string;
}

export interface QueueUrgencyNotificationSummary {
  badges: string[];
  reason: string;
  nextAction: string;
  evidenceSummary: string;
  schedulerState: Record<string, any> | null;
}

export interface FollowUpOutcomeNotificationSummary {
  badges: string[];
  outcome: string;
  summary: string;
}

export interface HypothesisReevaluationNotificationSummary {
  badges: string[];
  status: string;
  summary: string;
  sourceRunIds: string[];
  error: string;
}

export interface PolicyGuardrailNotificationSummary {
  badges: string[];
  action: string;
  reasons: string[];
}

export interface AutonomyBudgetNotificationSummary {
  badges: string[];
  throttleState: string;
  reasons: string[];
}

export interface CustomerAutonomyBudgetNotificationSummary {
  badges: string[];
  throttleState: string;
  reasons: string[];
}

function asExperimentRunNotificationData(data: Notification['data']): ExperimentRunNotificationData {
  return (data && typeof data === 'object') ? (data as ExperimentRunNotificationData) : {};
}

function asQueueUrgencyNotificationData(data: Notification['data']): QueueUrgencyNotificationData {
  return (data && typeof data === 'object') ? (data as QueueUrgencyNotificationData) : {};
}

function asFollowUpOutcomeNotificationData(data: Notification['data']): FollowUpOutcomeNotificationData {
  return (data && typeof data === 'object') ? (data as FollowUpOutcomeNotificationData) : {};
}

function asHypothesisReevaluationNotificationData(data: Notification['data']): HypothesisReevaluationNotificationData {
  return (data && typeof data === 'object') ? (data as HypothesisReevaluationNotificationData) : {};
}

function asAutonomyBudgetNotificationData(data: Notification['data']): AutonomyBudgetNotificationData {
  return (data && typeof data === 'object') ? (data as AutonomyBudgetNotificationData) : {};
}

function asCustomerAutonomyBudgetNotificationData(data: Notification['data']): CustomerAutonomyBudgetNotificationData {
  return (data && typeof data === 'object') ? (data as CustomerAutonomyBudgetNotificationData) : {};
}

export function summarizeExperimentNotification(notification: Notification): ExperimentNotificationSummary | null {
  if (notification.notification_type !== 'experiment_run_update') {
    return null;
  }

  const data = asExperimentRunNotificationData(notification.data);
  const badges: string[] = [];
  const finalPhase = String(data.final_phase || '').trim();
  const sourceName = String(data.source_name || '').trim();
  const sourceId = String(data.source_id || '').trim();
  const failedCommandCount = Number(data.failed_command_count || 0);
  const recoveryOpen = Boolean(data.recovery_open);
  const recoveryReason = String(data.recovery_reason || '').trim();
  const recommendedAction = String(data.recommended_action || '').trim();
  const latestOperatorAction = String(data.latest_operator_action || '').trim();
  const latestOperatorStatusBefore = String(data.latest_operator_status_before || '').trim();
  const latestOperatorStatusAfter = String(data.latest_operator_status_after || '').trim();
  const latestOperatorNote = String(data.latest_operator_note || '').trim();
  const latestOperatorOutcome = String(data.latest_operator_outcome || '').trim().replace(/_/g, ' ');
  const latestOperatorOutcomeReason = String(data.latest_operator_outcome_reason || '').trim();

  if (finalPhase) {
    badges.push(`Phase ${finalPhase}`);
  }
  if (sourceName) {
    badges.push(`Repo ${sourceName}`);
  } else if (sourceId) {
    badges.push(`Repo ${sourceId}`);
  }
  if (data.fallback_attempted) {
    badges.push(data.fallback_ok ? 'Fallback ok' : 'Fallback attempted');
  } else if (data.bootstrap_attempted) {
    badges.push(data.bootstrap_ok ? 'Bootstrap ok' : 'Bootstrap attempted');
  }
  if (recoveryOpen) {
    badges.push('Recovery open');
  }
  if (failedCommandCount > 0) {
    badges.push(`Failed cmds ${failedCommandCount}`);
  }

  let latestOperator = '';
  if (latestOperatorAction) {
    latestOperator = latestOperatorAction.replace(/_/g, ' ');
    if (latestOperatorStatusBefore || latestOperatorStatusAfter) {
      latestOperator += ` (${latestOperatorStatusBefore || '?'} -> ${latestOperatorStatusAfter || '?'})`;
    }
  }

  return {
    badges,
    reason: recoveryReason,
    nextAction: recommendedAction,
    latestOperator,
    latestOperatorNote,
    latestOperatorOutcome,
    latestOperatorOutcomeReason,
  };
}

export function summarizeQueueUrgencyNotification(notification: Notification): QueueUrgencyNotificationSummary | null {
  if (notification.notification_type !== 'queue_urgency_alert') {
    return null;
  }

  const data = asQueueUrgencyNotificationData(notification.data);
  const badges: string[] = [];
  const queueItemType = String(data.queue_item_type || '').trim();
  const slaBucket = String(data.sla_bucket || '').trim();
  const escalationLevel = String(data.escalation_level || '').trim();
  const customer = String(data.customer || '').trim();
  const ageMinutes = Number(data.age_minutes || 0);
  const priorityScore = Number(data.priority_score || 0);
  const schedulerState = data.scheduler_state && typeof data.scheduler_state === 'object' && !Array.isArray(data.scheduler_state)
    ? (data.scheduler_state as Record<string, any>)
    : null;

  if (queueItemType) badges.push(queueItemType.replace(/_/g, ' '));
  if (slaBucket) badges.push(slaBucket.replace(/_/g, ' '));
  if (escalationLevel) badges.push(`Esc ${escalationLevel}`);
  if (customer) badges.push(`Customer ${customer}`);
  if (data.is_stale) badges.push('Stale');
  if (ageMinutes > 0) badges.push(`Age ${ageMinutes}m`);
  if (priorityScore > 0) badges.push(`Urgency ${priorityScore}`);

  return {
    badges,
    reason: String(data.reason_label || '').trim(),
    nextAction: String(data.recommended_action || '').trim(),
    evidenceSummary: String(data.evidence_summary || '').trim(),
    schedulerState,
  };
}

export function summarizeFollowUpOutcomeNotification(notification: Notification): FollowUpOutcomeNotificationSummary | null {
  if (notification.notification_type !== 'follow_up_outcome_alert') {
    return null;
  }

  const data = asFollowUpOutcomeNotificationData(notification.data);
  const badges: string[] = [];
  const outcome = String(data.follow_up_outcome_status || '').trim();
  const recommendationKey = String(data.follow_up_recommendation_key || '').trim();
  const customer = String(data.customer || '').trim();
  const policyMode = String(data.follow_up_policy_mode || '').trim();
  const originSourceKind = String(data.origin_source_kind || '').trim();

  if (outcome) badges.push(outcome.replace(/_/g, ' '));
  if (recommendationKey) badges.push(recommendationKey.replace(/_/g, ' '));
  if (originSourceKind === 'profile') badges.push('Profile follow-up');
  if (originSourceKind === 'portfolio') badges.push('Fleet follow-up');
  if (customer) badges.push(`Customer ${customer}`);
  if (policyMode) badges.push(`Policy ${policyMode.replace(/_/g, ' ')}`);

  return {
    badges,
    outcome: outcome.replace(/_/g, ' '),
    summary: String(data.follow_up_outcome_summary || '').trim(),
  };
}

export function summarizeHypothesisReevaluationNotification(notification: Notification): HypothesisReevaluationNotificationSummary | null {
  if (notification.notification_type !== 'hypothesis_reevaluation_update') {
    return null;
  }

  const data = asHypothesisReevaluationNotificationData(notification.data);
  const badges: string[] = [];
  const status = String(data.reevaluation_status || '').trim().toLowerCase();
  const sourceRunIds = Array.isArray(data.source_run_ids)
    ? data.source_run_ids.map((value) => String(value || '').trim()).filter(Boolean)
    : [];
  const originSourceKind = String(data.origin_source_kind || '').trim().toLowerCase();
  if (status) badges.push(status.replace(/_/g, ' '));
  if (sourceRunIds.length > 0) badges.push(`Runs ${sourceRunIds.slice(0, 3).join(', ')}`);
  if (originSourceKind === 'profile') badges.push('Domain opportunity');
  if (originSourceKind === 'portfolio') badges.push('Fleet opportunity');

  return {
    badges,
    status,
    summary: String(data.reprioritization_summary || '').trim(),
    sourceRunIds,
    error: String(data.reevaluation_error || '').trim(),
  };
}

export function summarizePolicyGuardrailNotification(notification: Notification): PolicyGuardrailNotificationSummary | null {
  if (notification.notification_type !== 'policy_guardrail_alert') {
    return null;
  }
  const data = (notification.data || {}) as any;
  const action = String(data.policy_guardrail_action || '').trim();
  const customer = String(data.customer || '').trim();
  const reasons = Array.isArray(data.policy_guardrail_reasons)
    ? data.policy_guardrail_reasons.map((value: any) => String(value || '').trim()).filter(Boolean)
    : [];
  const badges: string[] = ['policy safeguard'];
  if (action) badges.push(action.replace(/_/g, ' '));
  if (customer) badges.push(`Customer ${customer}`);
  return {
    badges,
    action: action.replace(/_/g, ' '),
    reasons,
  };
}

export function summarizeAutonomyBudgetNotification(notification: Notification): AutonomyBudgetNotificationSummary | null {
  if (notification.notification_type !== 'autonomy_budget_alert') {
    return null;
  }
  const data = asAutonomyBudgetNotificationData(notification.data);
  const throttleState = String(data.budget_throttle_state || '').trim();
  const customer = String(data.customer || '').trim();
  const reasons = Array.isArray(data.budget_throttle_reasons)
    ? data.budget_throttle_reasons.map((value: any) => String(value || '').trim()).filter(Boolean)
    : [];
  const badges: string[] = ['budget throttle'];
  if (throttleState) badges.push(throttleState.replace(/_/g, ' '));
  if (customer) badges.push(`Customer ${customer}`);
  return {
    badges,
    throttleState: throttleState.replace(/_/g, ' '),
    reasons,
  };
}

export function summarizeCustomerAutonomyBudgetNotification(notification: Notification): CustomerAutonomyBudgetNotificationSummary | null {
  if (notification.notification_type !== 'customer_autonomy_budget_alert') {
    return null;
  }
  const data = asCustomerAutonomyBudgetNotificationData(notification.data);
  const throttleState = String(data.customer_budget_throttle_state || '').trim();
  const customer = String(data.customer || '').trim();
  const reasons = Array.isArray(data.customer_budget_throttle_reasons)
    ? data.customer_budget_throttle_reasons.map((value: any) => String(value || '').trim()).filter(Boolean)
    : [];
  const badges: string[] = ['customer budget throttle'];
  if (customer) badges.push(`Customer ${customer}`);
  if (throttleState) badges.push(throttleState.replace(/_/g, ' '));
  return {
    badges,
    throttleState: throttleState.replace(/_/g, ' '),
    reasons,
  };
}

export function buildNotificationToastSummary(notification: Notification): NotificationToastSummary {
  const experimentSummary = summarizeExperimentNotification(notification);
  const queueSummary = summarizeQueueUrgencyNotification(notification);
  const followUpSummary = summarizeFollowUpOutcomeNotification(notification);
  const reevaluationSummary = summarizeHypothesisReevaluationNotification(notification);
  const policyGuardrailSummary = summarizePolicyGuardrailNotification(notification);
  const autonomyBudgetSummary = summarizeAutonomyBudgetNotification(notification);
  const customerAutonomyBudgetSummary = summarizeCustomerAutonomyBudgetNotification(notification);
  if (!experimentSummary && !queueSummary && !followUpSummary && !reevaluationSummary && !policyGuardrailSummary && !autonomyBudgetSummary && !customerAutonomyBudgetSummary) {
    return {
      title: notification.title,
      description: String(notification.message || '').trim(),
    };
  }

  if (queueSummary) {
    const descriptionParts: string[] = [];
    if (queueSummary.badges.length > 0) {
      descriptionParts.push(queueSummary.badges.slice(0, 4).join(' · '));
    }
    if (queueSummary.reason) {
      descriptionParts.push(`Reason: ${queueSummary.reason}`);
    }
    if (queueSummary.nextAction) {
      descriptionParts.push(`Next: ${queueSummary.nextAction}`);
    }
    if (queueSummary.evidenceSummary) {
      descriptionParts.push(`Evidence: ${queueSummary.evidenceSummary}`);
    }

    return {
      title: notification.title,
      description: descriptionParts.join('\n').trim() || String(notification.message || '').trim(),
    };
  }

  if (followUpSummary) {
    const descriptionParts: string[] = [];
    if (followUpSummary.badges.length > 0) {
      descriptionParts.push(followUpSummary.badges.slice(0, 4).join(' · '));
    }
    if (followUpSummary.summary) {
      descriptionParts.push(`Summary: ${followUpSummary.summary}`);
    }
    return {
      title: notification.title,
      description: descriptionParts.join('\n').trim() || String(notification.message || '').trim(),
    };
  }

  if (reevaluationSummary) {
    const descriptionParts: string[] = [];
    if (reevaluationSummary.badges.length > 0) {
      descriptionParts.push(reevaluationSummary.badges.slice(0, 4).join(' · '));
    }
    if (reevaluationSummary.summary) {
      descriptionParts.push(`Summary: ${reevaluationSummary.summary}`);
    }
    if (reevaluationSummary.error) {
      descriptionParts.push(`Error: ${reevaluationSummary.error}`);
    }
    return {
      title: notification.title,
      description: descriptionParts.join('\n').trim() || String(notification.message || '').trim(),
    };
  }

  if (policyGuardrailSummary) {
    const descriptionParts: string[] = [];
    if (policyGuardrailSummary.badges.length > 0) {
      descriptionParts.push(policyGuardrailSummary.badges.join(' · '));
    }
    if (policyGuardrailSummary.reasons.length > 0) {
      descriptionParts.push(`Why: ${policyGuardrailSummary.reasons.join(' · ')}`);
    }
    return {
      title: notification.title,
      description: descriptionParts.filter(Boolean).join('\n'),
    };
  }

  if (autonomyBudgetSummary) {
    const descriptionParts: string[] = [];
    if (autonomyBudgetSummary.badges.length > 0) {
      descriptionParts.push(autonomyBudgetSummary.badges.join(' · '));
    }
    if (autonomyBudgetSummary.reasons.length > 0) {
      descriptionParts.push(`Why: ${autonomyBudgetSummary.reasons.join(' · ')}`);
    }
    return {
      title: notification.title,
      description: descriptionParts.filter(Boolean).join('\n'),
    };
  }

  if (customerAutonomyBudgetSummary) {
    const descriptionParts: string[] = [];
    if (customerAutonomyBudgetSummary.badges.length > 0) {
      descriptionParts.push(customerAutonomyBudgetSummary.badges.slice(0, 4).join(' · '));
    }
    if (customerAutonomyBudgetSummary.reasons.length > 0) {
      descriptionParts.push(customerAutonomyBudgetSummary.reasons.slice(0, 3).join(' · '));
    }
    return {
      title: notification.title,
      description: descriptionParts.join('\n').trim() || String(notification.message || '').trim(),
    };
  }

  const descriptionParts: string[] = [];
  const resolvedExperimentSummary = experimentSummary || {
    badges: [],
    reason: '',
    nextAction: '',
    latestOperator: '',
    latestOperatorOutcome: '',
    latestOperatorOutcomeReason: '',
  };
  const compactBadges = resolvedExperimentSummary.badges.filter((badge) => badge !== 'Recovery open').slice(0, 3);
  if (compactBadges.length > 0) {
    descriptionParts.push(compactBadges.join(' · '));
  }
  if (resolvedExperimentSummary.reason) {
    descriptionParts.push(`Reason: ${resolvedExperimentSummary.reason}`);
  }
  if (resolvedExperimentSummary.nextAction) {
    descriptionParts.push(`Next: ${resolvedExperimentSummary.nextAction}`);
  }
  if (resolvedExperimentSummary.latestOperator) {
    descriptionParts.push(`Last operator: ${resolvedExperimentSummary.latestOperator}`);
  }
  if (resolvedExperimentSummary.latestOperatorOutcome) {
    descriptionParts.push(`Operator outcome: ${resolvedExperimentSummary.latestOperatorOutcome}`);
  }
  if (resolvedExperimentSummary.latestOperatorOutcomeReason) {
    descriptionParts.push(`Outcome reason: ${resolvedExperimentSummary.latestOperatorOutcomeReason}`);
  }

  return {
    title: notification.title,
    description: descriptionParts.join('\n').trim() || String(notification.message || '').trim(),
  };
}
