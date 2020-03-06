(define (domain construction)
  (:requirements :strips :equality)
  (:predicates
    (Robot ?r)
    (Node ?n)
    (Element ?e)
    (Printed ?e)
    (Removed ?e)
    (Traj ?r ?t)
    (PrintAction ?r ?n1 ?e ?n2 ?q1 ?q2 ?t)
    (Collision ?t ?e)
    (Grounded ?n)
    (Direction ?n1 ?e ?n2)
    (Order ?e1 ?e2)
    (Stiff)
    (Conf ?r ?q)
    (AtConf ?r ?q)
    (CanMove ?r)
    (CFreeTrajConf ?r ?t ?r2 ?q2)
    (UnsafeTraj ?r ?t)
  )

  ;(:action move
  ;  :parameters (?r ?q1 ?q2)
  ;  :precondition (and (Conf ?r ?q1) (Conf ?r ?q2)
  ;                     (AtConf ?r ?q1) (CanMove ?r))
  ;  :effect (and (AtConf ?r ?q2)
  ;               (not (AtConf ?r ?q1)) (not (CanMove ?r)))
  ;)

  (:action print
    :parameters (?r ?n1 ?e ?n2 ?q1 ?q2 ?t)
    :precondition (and (PrintAction ?r ?n1 ?e ?n2 ?q1 ?q2 ?t) (Printed ?e) ; (Stiff)
                       ; (AtConf ?r ?q1) ; (not (UnsafeTraj ?r ?t))
                       (forall (?e2) (imply (Order ?e ?e2) (Removed ?e2)))
                       (forall (?e2) (imply (Collision ?t ?e2) (Removed ?e2)))
                  )
    :effect (and (Removed ?e) (CanMove ?r)
                 (AtConf ?r ?q2) ; TODO: reverse?
                 (not (AtConf ?r ?q1))
                 (not (Printed ?e)))
  )

  ;(:derived (UnsafeTraj ?r ?t) (and
  ;      (Traj ?r ?t)
  ;      (exists (?r2 ?q2) (and (Conf ?r2 ?q2) (not (= ?r ?r2))
  ;                             (not (CFreeTrajConf ?r ?t ?r2 ?q2))
  ;                             (AtConf ?r2 ?q2))))
  ;)
)