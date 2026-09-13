; Obligation 1 -- is the fused select equivalent to the pair it replaces?
;
; The pair is Godot's branchless clamp shape, the hottest fusable sequence in
; core/math: `fcmgt` builds a lane mask of all-ones or all-zeros from a float
; compare, then `bit` inserts under that mask. Fusing them means doing the
; compare and the selection in one instruction.
;
; The sequence is composed from the emitted semantics of the two constituents,
; so it is the description's own account of the pair rather than a restatement
; of what the pair is supposed to do.
;
; NaN is the whole question here. An unordered compare makes fcmgt produce a
; zero mask, so `bit` inserts nothing and the destination survives. Any fused
; form must agree with that, and the obvious alternative does not -- see
; nan_control.smt2.

(declare-const fd (_ BitVec 32))
(declare-const fn1 (_ BitVec 32))
(declare-const fm (_ BitVec 32))

(define-fun sequence ((fd (_ BitVec 32)) (fn1 (_ BitVec 32)) (fm (_ BitVec 32))) (_ BitVec 32)
  (axis_instr_bit_s fd fn1 (axis_instr_fcmgt_s fn1 fm)))

; Assert the NEGATION: unsat means no input distinguishes them.
(assert (not (= (axis_instr_fselgt fd fn1 fm) (sequence fd fn1 fm))))
(check-sat)
