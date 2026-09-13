; Obligation 1 -- does the AXIS description mean what it claims?
;
; Written independently of the .axisl: this states, lane by lane and with
; explicit extracts, what SMLALB is supposed to compute -- take the low signed
; byte of each 32-bit lane of vn and vm, multiply them, add to that lane of vd.
; It deliberately uses sign_extend/extract rather than the shift pair the AXIS
; source uses, so that agreement means the two say the same thing rather than
; that the same expression was written twice.
;
; This is the obligation that catches a per-lane shift amount which is really a
; whole-register constant: with a plain `24` in the AXIS source, lanes 1-3 do
; not widen at all and this returns sat.

(define-fun lane_ref ((vd (_ BitVec 128)) (vn (_ BitVec 128)) (vm (_ BitVec 128))) (_ BitVec 128)
  (concat
    (bvadd ((_ extract 127 96) vd)
           (bvmul ((_ sign_extend 24) ((_ extract 103 96) vn))
                  ((_ sign_extend 24) ((_ extract 103 96) vm))))
    (concat
      (bvadd ((_ extract 95 64) vd)
             (bvmul ((_ sign_extend 24) ((_ extract 71 64) vn))
                    ((_ sign_extend 24) ((_ extract 71 64) vm))))
      (concat
        (bvadd ((_ extract 63 32) vd)
               (bvmul ((_ sign_extend 24) ((_ extract 39 32) vn))
                      ((_ sign_extend 24) ((_ extract 39 32) vm))))
        (bvadd ((_ extract 31 0) vd)
               (bvmul ((_ sign_extend 24) ((_ extract 7 0) vn))
                      ((_ sign_extend 24) ((_ extract 7 0) vm))))))))

(declare-const vd (_ BitVec 128))
(declare-const vn (_ BitVec 128))
(declare-const vm (_ BitVec 128))

; Assert the NEGATION: unsat means no input distinguishes them.
(assert (not (= (axis_instr_smlalb vd vn vm) (lane_ref vd vn vm))))
(check-sat)
