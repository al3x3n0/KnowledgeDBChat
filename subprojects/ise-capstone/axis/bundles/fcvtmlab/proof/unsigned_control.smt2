; Obligation 2 -- does the check have teeth?
;
; The same conversion reading the byte as unsigned. INT8 activations are
; signed, so this is wrong, and it is wrong in a way that a benchmark on data
; that happens to be mostly non-negative would not reveal.

(declare-const fn1 (_ BitVec 32))

(assert (not (= (axis_instr_cvtb_u_bug fn1)
                (axis_instr_scvtf_s (axis_instr_sxtl_s fn1)))))
(check-sat)
(get-value (fn1))
