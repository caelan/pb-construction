(define (stream construction)
  ;(:stream test-cfree
  ;  :inputs (?t ?e)
  ;  :domain (and (Traj ?t) (Element ?e))
  ;  :certified (CFree ?t ?e)
  ;)
  (:stream sample-print
    :inputs (?r ?n ?e)
    :domain (and (Robot ?r) (StartNode ?n ?e))
    ; :fluents (Printed)
    :outputs (?q1 ?q2 ?t)
    :certified (and (PrintAction ?r ?n ?e ?q1 ?q2 ?t)
                    (Conf ?r ?q1) (Conf ?r ?q2) (Traj ?t))
  )
  (:stream test-stiffness
   :fluents (Printed)
   :certified (Stiff)
  )
)