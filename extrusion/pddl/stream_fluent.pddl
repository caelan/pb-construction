(define (stream construction)

  ;(:stream sample-move
  ;  :inputs (?r ?n1 ?n2)
  ;  :domain (and (Transition ?r ?n1 ?n2) (Move))
  ;  :outputs (?t)
  ;  :certified (and (MoveAction ?r ?n1 ?n2 ?t) ; Not well defined because no n1 n2
  ;                  (Traj ?r ?t))
  ;)

  (:stream sample-print
    :inputs (?r ?n1 ?e ?n2)
    :domain (and (Robot ?r) (Direction ?n1 ?e ?n2) (Assigned ?r ?e) (Print))
    ;:fluents (Printed)
    :outputs (?t)
    :certified (and (PrintAction ?r ?n1 ?e ?n2 ?t)
                    (Traj ?r ?t))
  )

  ;(:function (Distance ?n1 ?n2)
  ;           (and (Node ?n1) (Node ?n2)))
)