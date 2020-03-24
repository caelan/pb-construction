(define (domain extrusion)
  (:requirements :equality :typing :durative-actions :numeric-fluents :derived-predicates
                 :conditional-effects :disjunctive-preconditions)
  (:predicates
    (Robot ?r)
    (Node ?n)
    (Element ?e)
    (Printed ?e)
    (Removed ?e)
    (Traj ?r ?t)
    (MoveAction ?r ?q1 ?q2 ?t)
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
    (CFreeTrajTraj ?r1 ?t1 ?r2 ?t2)
    (UnsafeTraj ?r ?t)
    (Idle ?r)
    (Executing ?r ?t)
    (Assigned ?r ?e)
  )
  (:functions
    (Speed)
    (Length ?e)
    (Distance ?r ?t)
  )

  (:durative-action move
   :parameters (?r ?q1 ?q2 ?t)
   :duration (= ?duration (/ (Distance ?r ?t) (Speed))) ; TODO: trajectory time
   :condition (and
        (at start (MoveAction ?r ?q1 ?q2 ?t))
        (at start (imply (Sequential) (Stationary)))
        (at start (Idle ?r))
        (at start (AtConf ?r ?q2))
        (over all (not (UnsafeTraj ?r ?t)))
        ;(CanMove ?r)
   )
   :effect (and
        (at start (not (Stationary)))
        (at start (not (Idle ?r)))
        (at start (Executing ?r ?t))
        (at start (not (AtConf ?r ?q2)))
        (at end (AtConf ?r ?q1))
        (at end (not (Executing ?r ?t)))
        (at end (Idle ?r))
        (at end (Stationary))
   )
  )

  ; TODO: could sample multiple speeds
  (:durative-action print
   :parameters (?r ?n1 ?e ?n2 ?q1 ?q2 ?t)
   ;:duration (= ?duration (/ (Length ?e) (Speed)))
   :duration (= ?duration (/ (Distance ?r ?t) (Speed)))
   :condition (and
        (at start (PrintAction ?r ?n1 ?e ?n2 ?q1 ?q2 ?t))
        (at start (Assigned ?r ?e))
        (at start (imply (Sequential) (Stationary)))
        (at start (Printed ?e))
        (at start (Idle ?r))
        (at start (imply (Move) (AtConf ?r ?q2)))
        ; TODO: remove universal quantifiers to reschedule
        ;(at start (not (Premature ?e))) ; normalized to be a universal quantifier
        (at start (forall (?e2) (imply (Order ?e ?e2) (Removed ?e2))))
        (at start (forall (?e2) (imply (Collision ?t ?e2) (Removed ?e2))))
        (over all (not (UnsafeTraj ?r ?t)))
   )
   :effect (and
        (at start (not (Stationary)))
        (at start (not (Idle ?r)))
        (at start (not (Printed ?e)))
        (at start (Executing ?r ?t))
        (at start (not (AtConf ?r ?q2)))
        (at end (AtConf ?r ?q1))
        (at end (not (Executing ?r ?t)))
        (at end (Removed ?e))
        (at end (Idle ?r))
        (at end (Stationary))
   )
  )

  ;(:derived (Premature ?e)
  ;    (exists (?e2) (and (Order ?e ?e2)
  ;                       (not (Removed ?e2))))) ; Positive form instead

  (:derived (UnsafeTraj ?r1 ?t1) (and (CTraj ?r1 ?t1) (or
      ;(exists (?r2 ?q2) (and (Conf ?r2 ?q2) (not (= ?r1 ?r2))
      ;                       (TrajConfCollision ?r1 ?t1 ?r2 ?q2)
      ;                       (AtConf ?r2 ?q2)))
      (exists (?r2 ?t2) (and (CTraj ?r2 ?t2) (not (= ?r1 ?r2))
                             ;(not (CFreeTrajTraj ?r1 ?t1 ?r2 ?t2))
                             (TrajTrajCollision ?r1 ?t1 ?r2 ?t2)
                             (Executing ?r2 ?t2))))))

  ;(:derived (UnsafeConf ?r1 ?q1) ; TODO
)