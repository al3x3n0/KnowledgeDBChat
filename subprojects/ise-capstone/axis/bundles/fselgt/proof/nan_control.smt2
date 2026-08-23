; Obligation 2 -- does the equivalence check have teeth?
;
; An obligation that cannot fail is not evidence. FSELGT_NAN_BUG is the
; implementation a reasonable engineer reaches for first: instead of testing
; `fm < fn1`, logically negate `fn1 <= fm`. On ordinary numbers those agree.
; On a NaN they do not -- every ordered compare is false when an operand is
; NaN, so negating one yields true and the wrong operand is selected.
;
; This is worth proving rather than reasoning about, because a benchmark over
; real geometry would essentially never produce the NaN that separates them,
; and the two forms cost the same. Nothing except a proof distinguishes them.

(declare-const fd (_ BitVec 32))
(declare-const fn1 (_ BitVec 32))
(declare-const fm (_ BitVec 32))

(assert (not (= (axis_instr_fselgt_nan_bug fd fn1 fm)
                (axis_instr_bit_s fd fn1 (axis_instr_fcmgt_s fn1 fm)))))
(check-sat)
(get-value (fd fn1 fm))
