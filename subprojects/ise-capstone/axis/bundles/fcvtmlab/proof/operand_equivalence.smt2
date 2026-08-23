; Obligation 1 -- is the fused conversion the same as the two-step one?
;
; This is the whole of what fusing sxtl+scvtf+fmla changes. All three forms end
; in the same `fma` applied to the same fm and fd; they differ only in how the
; float operand is produced -- the pair widens byte to word and then converts,
; the fused instruction converts straight from the byte. `fma` is a function of
; its arguments, so equal operands give equal results and this settles the
; whole instruction by substitution.
;
; It is stated at this level for a reason beyond tidiness. The undecomposed
; obligation -- all three operands symbolic under an IEEE-754 fma -- does not
; terminate in a usable time, while this one is instant. Where a proof can be
; reduced to the part that actually differs, it should be.

(declare-const fn1 (_ BitVec 32))

; Assert the NEGATION: unsat means no byte value distinguishes them.
(assert (not (= (axis_instr_cvtb_s fn1)
                (axis_instr_scvtf_s (axis_instr_sxtl_s fn1)))))
(check-sat)
