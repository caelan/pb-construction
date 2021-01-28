(define (domain extrusion-backward)
  (:requirements :strips :equality)
  (:predicates
    (Robot ?r)
    (Node ?n)
    (Element ?e)
    (Endpoint ?n1 ?e)
    (Edge ?n1 ?e ?n2)
    (Location ?l)

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
    (Stiff)

    ; Derived
    (Connected ?n)
    (Printable ?n ?e) ; ?n is connected even when ?e is removed
  )
  (:functions
    (LocationDistance ?l1 ?l2)
  )

  ;(:action move ; -backward
  ;  :parameters (?r ?l1 ?l2)
  ;  :precondition (and (Robot ?r) (Location ?l1) (Location ?l2) (not (= ?l1 ?l2)) (Movable ?r)
  ;                     (AtNode ?r ?l2) (CanMove ?r)) ; TODO: rename to location or waypoint
  ;  :effect (and (AtNode ?r ?l1)
  ;               (not (AtNode ?r ?l2)) (not (CanMove ?r))
  ;               (increase (total-cost) (LocationDistance ?l1 ?l2)))
  ;)

  (:action print ; -backward
    :parameters (?r ?n1 ?e ?n2 ?t)
    :precondition (and (PrintAction ?r ?n1 ?e ?n2 ?t) (Printed ?e)
                       ; (AtNode ?r ?n2)
                       (Stiff)
                       (Printable ?n1 ?e)
                       ;(or (Grounded ?n1)
                       ;    (exists (?n0 ?e0) (and (Edge ?n0 ?e0 ?n1) (not (= ?e0 ?e)) ; (not (= ?n0 ?n2))
                       ;                           (Connected ?n0) (Printed ?e0)
                       ;)))
                       ;(forall (?e2) (imply (Order ?e ?e2) (Removed ?e2)))
                  )
    :effect (and (Removed ?e) (CanMove ?r)
                 (AtNode ?r ?n1)
                 (not (AtNode ?r ?n2))
                 (not (Printed ?e))
                 (increase (total-cost) 1)
            ))

  (:derived (Connected ?n2)
    (or (Grounded ?n2)
        (exists (?n1 ?e) (and (Edge ?n1 ?e ?n2)
                              (Connected ?n1) (Printed ?e)
                         ))))

  (:derived (Printable ?n1 ?e) (and (Endpoint ?n1 ?e)
    (or (Grounded ?n1)
      (exists (?n0 ?e0) (and (Edge ?n0 ?e0 ?n1) (not (= ?e0 ?e)) ; (not (= ?n0 ?n2))
                             (Connected ?n0) (Printed ?e0)
    )))
  ))
)