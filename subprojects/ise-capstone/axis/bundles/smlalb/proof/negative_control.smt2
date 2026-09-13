; Obligation 3 -- does the equivalence check have teeth?
;
; An obligation that cannot fail is not evidence, and the two unsat results
; above are only worth something if this one comes back sat. SMLALB_ZEXT_BUG is
; the same instruction with unsigned widening -- the mistake an INT8 pipeline
; could plausibly make, and one that a benchmark on mostly-positive activations
; would very likely miss.

(declare-const vd (_ BitVec 128))
(declare-const vn (_ BitVec 128))
(declare-const vm (_ BitVec 128))

(assert (not (= (axis_instr_smlalb_zext_bug vd vn vm)
                (axis_instr_smlal_h_s vd (axis_instr_sxtl_b_h vn) (axis_instr_sxtl_b_h vm)))))
(check-sat)
(get-value (vd vn vm))
