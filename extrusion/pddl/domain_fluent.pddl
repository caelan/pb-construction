(define (domain extrusion-backward)
  (:requirements :strips :equality)
  (:predicates
    (Robot ?r)
    (Node ?n)
    (Element ?e)

    (Traj ?r ?t)
    (PrintAction ?r ?n1 ?e ?n2 ?t) ; print ?e from ?n1 to ?n2
    (Grounded ?n)
    (Direction ?n1 ?e ?n2)
    (Order ?e1 ?e2)
    (Assigned ?r ?e)

    ; Fluent
    (Printed ?e)
    (Removed ?e)
    (AtNode ?r ?q)
    (CanMove ?r)
    (Connected)
    (Stiff)

    ; Derived
    (Connected ?n ?e) ; ?n is connected even when ?e is removed
  )
  (:functions
    (NodeDistance ?n1 ?n2)
  )

  ;(:action move ; -backward
  ;  :parameters (?r ?n1 ?n2)
  ;  :precondition (and (Robot ?r) (Node ?n1) (Node ?n2) (not (= ?n1 ?n2))
  ;                     (AtNode ?r ?n2) (CanMove ?r))
  ;  :effect (and (AtNode ?r ?n1)
  ;               (not (AtNode ?r ?n2)) (not (CanMove ?r))
  ;               (increase (total-cost) (NodeDistance ?n1 ?n2)))
  ;)

  (:action print ; -backward
    :parameters (?r ?n1 ?e ?n2 ?t)
    :precondition (and (PrintAction ?r ?n1 ?e ?n2 ?t) (Printed ?e)
                       (Connected ?n1 ?e)
                       (Stiff)
                       ; (AtNode ?r ?n2)
                       ;(forall (?e2) (imply (Order ?e ?e2) (Removed ?e2)))
                  )
    :effect (and (Removed ?e) (CanMove ?r)
                 (AtNode ?r ?n1)
                 (not (AtNode ?r ?n2))
                 (not (Printed ?e))
                 (increase (total-cost) 1)
            )
  )

  ;(:derived (Connected ?n ?e)
  ;  (or (Grounded ?n)
  ;      (exists (?n2 ?e2) (and (Edge ?n2 ?e2 ?n) ; Edge dones
  ;                             (Connected ?n2 ?e) (Printed ?e2)
  ;
  ;                             ))) ; Can also just do on StartNode
  ;)
)