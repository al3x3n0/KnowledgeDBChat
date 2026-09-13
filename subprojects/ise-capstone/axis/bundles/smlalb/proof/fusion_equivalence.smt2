; Obligation 2 -- is the fused instruction equivalent to the pair it replaces?
;
; This is the claim a proposal to Arm actually rests on. A cycle count argues
; the fused form is faster; this argues it computes the same thing, for every
; input, with no appeal to a benchmark or a sample of values.
;
; The sequence is composed from the emitted semantics of the two constituent
; instructions -- sxtl_b_h applied to each source, then smlal_h_s over the
; results -- so it is the description's own account of the pair, not a
; restatement of it.
;
; The substantive question is whether collapsing two widening steps into one
; loses anything: the pair widens byte -> halfword -> word with an intermediate
; register, the fused form widens byte -> word in one go.

(declare-const vd (_ BitVec 128))
(declare-const vn (_ BitVec 128))
(declare-const vm (_ BitVec 128))

(define-fun sequence ((vd (_ BitVec 128)) (vn (_ BitVec 128)) (vm (_ BitVec 128))) (_ BitVec 128)
  (axis_instr_smlal_h_s vd (axis_instr_sxtl_b_h vn) (axis_instr_sxtl_b_h vm)))

; Assert the NEGATION: unsat means no input distinguishes them.
(assert (not (= (axis_instr_smlalb vd vn vm) (sequence vd vn vm))))
(check-sat)
