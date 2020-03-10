(define (stream construction)
  ;(:stream test-cfree
  ;  :inputs (?t ?e)
  ;  :domain (and (Traj ?r ?t) (Element ?e))
  ;  :certified (CFree ?t ?e)
  ;)

  ;(:stream test-cfree-traj-conf
  ;  :inputs (?r ?t ?r2 ?q2)
  ;  :domain (and (Traj ?r ?t) (Conf ?r2 ?q2))
  ;  :certified (CFreeTrajConf ?r ?t ?r2 ?q2)
  ;)

  ;(:stream test-cfree-traj-traj
  ;  :inputs (?r1 ?t1 ?r2 ?t2)
  ;  :domain (and (Traj ?r1 ?t1) (Traj ?r2 ?t2))
  ;  :certified (CFreeTrajTraj ?r1 ?t1 ?r2 ?t2)
  ;)

  (:stream sample-print
    :inputs (?r ?n1 ?e ?n2)
    :domain (and (Robot ?r) (Direction ?n1 ?e ?n2) (Assigned ?r ?e))
    ; :fluents (Printed)
    :outputs (?q1 ?q2 ?t)
    :certified (and (PrintAction ?r ?n1 ?e ?n2 ?q1 ?q2 ?t)
                    (Conf ?r ?q1) (Conf ?r ?q2) (Traj ?r ?t))
  )

  (:predicate (TrajTrajCollision ?r1 ?t1 ?r2 ?t2)
     (and (Traj ?r1 ?t1) (Traj ?r2 ?t2))
  )

  ;(:function (Length ?e)
  ;  (and (Element ?e))
  ;)
  (:function (Distance ?r ?t)
    (Traj ?r ?t)
  )
)