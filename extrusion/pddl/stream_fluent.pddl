(define (stream construction)

  (:stream sample-print
    :inputs (?r ?n1 ?e ?n2)
    :domain (and (Robot ?r) (Direction ?n1 ?e ?n2) (Assigned ?r ?e) (Print))
    :fluents (Printed)
    :outputs (?t)
    :certified (PrintAction ?r ?n1 ?e ?n2 ?t)
  )

  ;(:stream test-printable
  ;  :inputs (?n ?e)
  ;  :domain (Endpoint ?n ?e)
  ;  :fluents (Printed)
  ;  :certified (Printable ?n ?e)
  ;)

  (:stream test-stiff
    :fluents (Printed)
    :certified (Stiff)
  )

  (:function (LocationDistance ?l1 ?l2)
    (and (Location ?l1) (Location ?l2)))
)