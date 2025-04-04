#lang rosette/safe

(require "./bounded.rkt")
(require rosette/lib/angelic rosette/lib/match rosette/lib/synthax)

(provide (all-defined-out))

; list functions that have reasonable defaults rather than throwing errors

(define (list-ref-noerr l i)
  (if (&&  (>= i 0) (< i (length l))) (list-ref l i)
      0))

(define (list-tail-noerr l i)
  (if (&& (>= i 0) (<= i (length l))) (list-tail l i)
      (list)))

(define (list-append l i)
  (append l (list i)))

(define (list-empty)
  (list))

(define (list-prepend i l)
  (cons i l))

(define (list-take-noerr l i)
  (if (<= i 0) (list) (if (&& (>= i 0) (< i (length l))) (take l i) l )))

(define (vec-slice-noerr l start end)
  (list-tail-noerr (list-take-noerr l end) start)
)

(define (vec-slice-with-length-noerr l start lst_length)
  (vec-slice-noerr l start (+ start lst_length))
)

(define (list-concat l1 l2)
  (append l1 l2))

; tuple functions

(define (make-tuple e1 . es)
  (cons e1 es))

(define (tupleGet t i)
  (list-ref-noerr t i))

; set functions

(define (set-singleton v)
  (list v))

(define (set-create)
  (list))

(define (set-eq s1 s2)
  (equal? s1 s2))

(define (set-union as bs)
  (match* (as bs)
    [((list) bs)  bs]
    [(as (list))  as]
    [((list a as ...) (list b bs ...))
     (if (equal? a b)
      (set-union (cons a as) bs)
      (if (< a b)
         (cons a (set-union as (cons b bs)))
         (cons b (set-union (cons a as) bs)))
     )]))

(define (set-member v s1)
  (ormap (lambda (c) (equal? v c)) s1))

(define (set-insert v s1)
  (set-union (set-singleton v) s1))

(define (set-subset s1 s2)
  (match* (s1 s2)
    [((list) bs)  #t]
    [(as (list))  #f]
    [((list a as ...) (list b bs ...))
     (if (equal? a b)
      (set-subset as bs)
      (if (< a b)
         #f
         (set-subset (cons a as) bs))
     )]))

(define (set-minus s1 s2)
  (match* (s1 s2)
    [((list) bs)  (list)]
    [(as (list))  as]
    [((list a as ...) (list b bs ...))
     (if (equal? a b)
      (set-minus as bs)
      (if (< a b)
         (cons a (set-minus as (cons b bs)))
         (set-minus (cons a as) bs))
     )]))

; map functions

(define (map-create)
  (list))

(define (map-normalize m)
  (match* (m)
    [((list)) (map-create)]
    [((list a as ...))
     (map-insert (map-normalize as) (car a) (cdr a) (lambda (a b) b))
     ]))

(define (map-eq s1 s2)
  (equal? s1 s2))

(define (getIdx m k)
    (if (equal? (length m) 0) 0 (if (equal? (tupleGet (matrix-ref-noerr m 0) 0) k) 0
        (+ 1 (getIdx (list-tail-noerr m 1) k)))))

(define (map-contains m k)
    (if (equal? (length m) 0) 0
        (if (equal? (tupleGet (matrix-ref-noerr m 0) 0) k) 1
            (map-contains (list-tail-noerr m 1) k))))

(define (map-insert m k v)
  (if (equal? (map-contains m k) 0)
    (append m (list (make-tuple k v)))
    (list-set m (getIdx m k) (make-tuple k v))))

(define (map-get m k)
    (if (equal? (length m) 0) 0 (if (equal? (tupleGet (matrix-ref-noerr m 0) 0) k) (tupleGet (matrix-ref-noerr m 0) 1)
        (map-get (list-tail-noerr m 1) k))))

; (define (map-union as bs value-merge)
;   (match* (as bs)
;     [((list) bs)  bs]
;     [(as (list))  as]
;     [((list a as ...) (list b bs ...))
;      (if (equal? (car a) (car b))
;       (cons (cons (car a) (value-merge (cdr a) (cdr b))) (map-union as bs value-merge))
;       (if (< (car a) (car b))
;          (cons a (map-union as (cons b bs) value-merge))
;          (cons b (map-union (cons a as) bs value-merge)))
;      )]))

; (define (map-get m k default)
;   (match* (m)
;     [((list)) default]
;     [((list a as ...))
;      (if (equal? (car a) k)
;       (cdr a)
;       (map-get as k default)
;      )]))

; (define (map-singleton k v)
;   (list (cons k v)))

;  (define (map-insert m k v value-merge)
;    (map-union (map-singleton k v) m value-merge))

; (define (mapInsert m k v)
;   (if (equal? (mapContains m k) 0)
;     (append m (list (make-tuple k v)))
;     (list-set m (getIdx m k) (make-tuple k v))))

; (define (map-minus m keys)
;   (match* (m keys)
;     [((list) bs)  (list)]
;     [(as (list))  as]
;     [((list a as ...) (list b bs ...))
;      (if (equal? (car a) b)
;       (map-minus as bs)
;       (if (< (car a) b)
;          (cons a (map-minus as (cons b bs)))
;          (map-minus (cons a as) bs))
;      )]))

; (define (map-values m)
;   (map (lambda (a) (cdr a)) m))

(define (matrix-ref-noerr l i)
  (if (&&  (>= i 0) (< i (length l))) (list-ref l i)
      (list)))

(define (matrix-tail-noerr l i)
  (if (&& (>= i 0) (<= i (length l))) (list-tail l i)
      (list)))

(define (matrix-row-slice-noerr l start end)
  (matrix-tail-noerr (matrix-take-noerr l end) start)
)

(define (matrix-row-slice-with-length-noerr l start lst_length)
  (matrix-row-slice-noerr l start (+ start lst_length))
)

(define (matrix-col-slice-noerr l start end)
  (if
    (< (matrix-length l) 1)
    (list)
    (matrix-prepend
      (vec-slice-noerr (matrix-ref-noerr l 0) start end)
      (matrix-col-slice-noerr (matrix-tail-noerr l 1) start end)
    )
  )
)

(define (matrix-col-slice-with-length-noerr l start lst_length)
  (matrix-col-slice-noerr l start (+ start lst_length))
)

(define (matrix-prepend i l)
  (if (list? i)
    (if (> (length i ) 0)
    (append (list i) l) (append l i)) (append (list i) l) ))

(define (matrix-take-noerr l i)
  (if (<= i 0) (list) (if (&& (>= i 0) (< i (length l))) (take l i) l )))

(define (matrix-length l) (length l))

(define (matrix-append l i)
(append l (list i)))

(define (matrix-empty)
  (list))

(define (list-tuple-ref-noerr l i)
  (if (&&  (>= i 0) (< i (length l))) (list-ref l i)
      (list)))
(define (list-tuple-tail-noerr l i)
  (if (&& (>= i 0) (<= i (length l))) (list-tail l i)
      (list)))
(define (list-tuple-empty)
(list))
(define (list-tuple-prepend i l)
  (append (list i) l))
(define (list-tuple-take-noerr l i)
  (if (<= i 0) (list) (if (&& (>= i 0) (< i (length l))) (take l i) l )))
(define (list-tuple-append l i)
(append l (list i)))
(define (list-tuple-length l) (length l))

(define (quotient-noerr x y)
  (quotient x (if (equal? y 0) 1 y))
)

; (define (transpose lst)
;   (if (< (length lst) 1) (list) (apply map list lst))
; )

(define-bounded (firsts matrix)
  (if (< (matrix-length matrix) 1)
      (list-empty)
      (list-prepend
        (list-ref-noerr (matrix-ref-noerr matrix 0) 0)
        (firsts (list-tail-noerr matrix 1))
      )
  )
)

(define (rests matrix)
  (if (< (matrix-length matrix) 1)
    (matrix-empty)
    (matrix-col-slice-noerr matrix 1 (length (matrix-ref-noerr matrix 0)))
  )
)

(define-bounded (matrix-transpose-noerr matrix)
  (if (< (matrix-length matrix) 1)
    (matrix-empty)
    (matrix-prepend (firsts matrix) (matrix-transpose-noerr (rests matrix)))
  )
)

(define-bounded (integer-sqrt-noerr n) n)

(define-bounded (integer-exp-noerr n)
  (if
    (<= n 0)
    1
    (remainder (* 3 (integer-exp-noerr (- n 1))) 64)
  )
)

; 3D tensors
(define (tensor3d-length l) (length l))

(define (tensor3d-empty)
  (list))

(define (tensor3d-ref-noerr l i)
  (if (&&  (>= i 0) (< i (length l))) (list-ref l i)
      (list)))

(define (tensor3d-tail-noerr l i)
  (if (&& (>= i 0) (<= i (length l))) (list-tail l i)
      (list)))

(define (tensor3d-take-noerr l i)
  (if (<= i 0) (list) (if (&& (>= i 0) (< i (length l))) (take l i) l )))

(define (tensor3d-slice-noerr l start end)
  (tensor3d-tail-noerr (tensor3d-take-noerr l end) start)
)

(define (tensor3d-slice-with-length-noerr l start lst_length)
  (tensor3d-slice-noerr l start (+ start lst_length))
)

(define (tensor3d-prepend i l)
  (if (list? i)
    (if (> (length i ) 0)
    (append (list i) l) (append l i)) (append (list i) l) ))

(define (tensor3d-append l i)
(append l (list i)))
