(define (stream construction)
  ;(:stream test-cfree
  ;  :inputs (?t ?e)
  ;  :domain (and (Traj ?r ?t) (Element ?e))
  ;  :certified (CFree ?t ?e)
  ;)

  (:stream test-cfree-traj-conf
    :inputs (?r ?t ?r2 ?q2)
    :domain (and (Traj ?r ?t) (Conf ?r2 ?q2))
    :certified (CFreeTrajConf ?r ?t ?r2 ?q2)
  )

  (:stream sample-print
    :inputs (?r ?n ?e)
    :domain (and (Robot ?r) (StartNode ?n ?e))
    ; :fluents (Printed)
    :outputs (?q1 ?q2 ?t)
    :certified (and (PrintAction ?r ?n ?e ?q1 ?q2 ?t)
                    (Conf ?r ?q1) (Conf ?r ?q2) (Traj ?r ?t))
  )

  (:stream test-stiffness
   :fluents (Printed)
   :certified (Stiff)
  )
)