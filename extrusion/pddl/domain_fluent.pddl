(define (domain construction)
  (:requirements :strips :equality)
  (:predicates
    (Robot ?r)
    (Node ?n)
    (Element ?e)
    (Edge ?n1 ?e ?n2)

    (Traj ?r ?t)
    (PrintAction ?r ?n1 ?e ?n2 ?t)
    (Grounded ?n)
    (Direction ?n1 ?e ?n2)
    (Order ?e1 ?e2)
    (Assigned ?r ?e)

    ; Fluent
    (Printed ?e)
    (Removed ?e)
    (Stiff)
    (AtNode ?r ?q)
    (CanMove ?r)

    ; Derived
    (Connected ?n)
  )
  (:functions
    (Distance ?n1 ?n2)
  )

  ;(:action move
  ;  :parameters (?r ?n1 ?n2)
  ;  :precondition (and (Robot ?r) (Node ?n1) (Node ?n2) (not (= ?n1 ?n2))
  ;                     (AtNode ?r ?n1) (CanMove ?r))
  ;  :effect (and (AtNode ?r ?n2)
  ;               (not (AtNode ?r ?n1)) (not (CanMove ?r))
  ;               (increase (total-cost) (Distance ?n1 ?n2)))
  ;)

  (:action print
    :parameters (?r ?n1 ?e ?n2 ?t)
    :precondition (and (PrintAction ?r ?n1 ?e ?n2 ?t) (Printed ?e) ; (Stiff)
                       ;(Connected ?n1) (Connected ?n2) ; No nodes disconnected
                       ; (AtNode ?r ?n1)
                       ;(forall (?e2) (imply (Order ?e ?e2) (Removed ?e2)))
                  )
    :effect (and (Removed ?e)
                 (AtNode ?r ?n2) (CanMove ?r)
                 (not (AtNode ?r ?n1))
                 (not (Printed ?e))
                 (increase (total-cost) 1)
            )
  )

  (:derived (Connected ?n2) ; TODO: define on nodes or elements
    (or (Grounded ?n2)
        (exists (?n1 ?e) (and (Edge ?n1 ?e ?n2)
                              (Printed ?e) (Connected ?n1)))) ; Can also just do on StartNode
  )
)