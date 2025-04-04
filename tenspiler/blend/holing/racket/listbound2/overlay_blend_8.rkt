#lang rosette
(require "./bounded.rkt")
(require "./utils.rkt")
(require rosette/lib/angelic rosette/lib/match rosette/lib/synthax)
(require rosette/solver/smt/bitwuzla)
(current-solver (bitwuzla #:path "/bitwuzla/build/src/main/bitwuzla" #:options (hash ':seed 0)))



 (define-bounded (selection_two_args x y select_two_args_arg)
(if (or (< (length x ) 1 ) (! (equal? (length x ) (length y ) ) ) ) (list-empty ) (list-prepend (select_two_args_arg (list-ref-noerr x 0 ) (list-ref-noerr y 0 )) (selection_two_args (list-tail-noerr x 1 ) (list-tail-noerr y 1 ) select_two_args_arg) ) ))


 (define-bounded (matrix_selection_two_args matrix_x matrix_y select_two_args_arg)
(if (or (< (matrix-length matrix_x ) 1 ) (! (equal? (matrix-length matrix_x ) (matrix-length matrix_y ) ) ) ) (matrix-empty ) (matrix-prepend (selection_two_args (matrix-ref-noerr matrix_x 0 ) (matrix-ref-noerr matrix_y 0 ) select_two_args_arg) (matrix_selection_two_args (matrix-tail-noerr matrix_x 1 ) (matrix-tail-noerr matrix_y 1 ) select_two_args_arg ) ) ))

(define-grammar (overlay_blend_8_inv0_gram active agg.result base col pixel row row_vec)
 [rv (choose (&& (&& (>= row 0 ) (<= row (matrix-length base ) ) ) (equal? agg.result (matrix_selection_two_args (v0) (v0) select_two_args ) ) ))]
[v0 (choose (if (OUTER_LOOP_INDEX_FIRST) (matrix-take-noerr (v1) row ) (matrix-col-slice-noerr (v1) 0 row ) ) (matrix-transpose-noerr (if (OUTER_LOOP_INDEX_FIRST) (matrix-take-noerr (v1) row ) (matrix-col-slice-noerr (v1) 0 row ) ) ))]
[v1 (choose base active)]
)

(define-grammar (overlay_blend_8_inv1_gram active base col pixel row_vec agg.result row)
 [rv (choose (&& (&& (&& (&& (&& (>= row 0 ) (<= row (matrix-length base ) ) ) (>= col 0 ) ) (<= col (length (matrix-ref-noerr base 0 ) ) ) ) (equal? row_vec (selection_two_args (if (OUTER_LOOP_INDEX_FIRST) (list-take-noerr (matrix-ref-noerr (v0) row ) col ) (matrix-ref-noerr (matrix-transpose-noerr (matrix-col-slice-with-length-noerr (matrix-take-noerr (v0) col ) row 1 ) ) 0 ) ) (if (OUTER_LOOP_INDEX_FIRST) (list-take-noerr (matrix-ref-noerr (v0) row ) col ) (matrix-ref-noerr (matrix-transpose-noerr (matrix-col-slice-with-length-noerr (matrix-take-noerr (v0) col ) row 1 ) ) 0 ) ) select_two_args) ) ) (equal? agg.result (matrix_selection_two_args (v1) (v1) select_two_args ) ) ))]
[v0 (choose base active)]
[v1 (choose (if (OUTER_LOOP_INDEX_FIRST) (matrix-take-noerr (v0) row ) (matrix-col-slice-noerr (v0) 0 row ) ) (matrix-transpose-noerr (if (OUTER_LOOP_INDEX_FIRST) (matrix-take-noerr (v0) row ) (matrix-col-slice-noerr (v0) 0 row ) ) ))]
)

(define-grammar (overlay_blend_8_ps_gram base active overlay_blend_8_rv)
 [rv (choose (equal? overlay_blend_8_rv (matrix_selection_two_args (v0) (v0) select_two_args ) ))]
[v0 (choose (v1) (matrix-transpose-noerr (v1) ))]
[v1 (choose base active)]
)

(define-grammar (select_two_args_gram int_x int_y)
 [rv (choose (if (>= (v0) (v1) ) (- (- (+ (* (v1) (v0) ) (v0) ) (quotient-noerr (* (* (v1) (v0) ) (v0) ) (v1) ) ) (v1) ) (quotient-noerr (* (* (v1) (v0) ) (v0) ) (v1) ) ))]
[v0 (choose int_x int_y)]
[v1 (choose 2 16 32)]
)

(define-grammar (OUTER_LOOP_INDEX_FIRST_gram )
 [rv (choose (v0))]
[v0 (choose true false)]
)

(define (overlay_blend_8_inv0 active agg.result base col pixel row row_vec) (overlay_blend_8_inv0_gram active agg.result base col pixel row row_vec #:depth 10))
(define (overlay_blend_8_inv1 active base col pixel row_vec agg.result row) (overlay_blend_8_inv1_gram active base col pixel row_vec agg.result row #:depth 10))
(define (overlay_blend_8_ps base active overlay_blend_8_rv) (overlay_blend_8_ps_gram base active overlay_blend_8_rv #:depth 10))

(define (OUTER_LOOP_INDEX_FIRST ) (OUTER_LOOP_INDEX_FIRST_gram  #:depth 10))
(define (select_two_args int_x int_y) (select_two_args_gram int_x int_y #:depth 10))

(define-symbolic active_BOUNDEDSET-len integer?)
(define-symbolic active_BOUNDEDSET-0 integer?)
(define-symbolic active_BOUNDEDSET-1 integer?)
(define-symbolic active_BOUNDEDSET-2 integer?)
(define-symbolic active_BOUNDEDSET-3 integer?)
(define active (take (list (list active_BOUNDEDSET-0 active_BOUNDEDSET-1) (list active_BOUNDEDSET-2 active_BOUNDEDSET-3)) active_BOUNDEDSET-len))
(define-symbolic agg.result_BOUNDEDSET-len integer?)
(define-symbolic agg.result_BOUNDEDSET-0 integer?)
(define-symbolic agg.result_BOUNDEDSET-1 integer?)
(define-symbolic agg.result_BOUNDEDSET-2 integer?)
(define-symbolic agg.result_BOUNDEDSET-3 integer?)
(define agg.result (take (list (list agg.result_BOUNDEDSET-0 agg.result_BOUNDEDSET-1) (list agg.result_BOUNDEDSET-2 agg.result_BOUNDEDSET-3)) agg.result_BOUNDEDSET-len))
(define-symbolic base_BOUNDEDSET-len integer?)
(define-symbolic base_BOUNDEDSET-0 integer?)
(define-symbolic base_BOUNDEDSET-1 integer?)
(define-symbolic base_BOUNDEDSET-2 integer?)
(define-symbolic base_BOUNDEDSET-3 integer?)
(define base (take (list (list base_BOUNDEDSET-0 base_BOUNDEDSET-1) (list base_BOUNDEDSET-2 base_BOUNDEDSET-3)) base_BOUNDEDSET-len))
(define-symbolic col integer?)
(define-symbolic int_x integer?)
(define-symbolic int_y integer?)
(define-symbolic overlay_blend_8_rv_BOUNDEDSET-len integer?)
(define-symbolic overlay_blend_8_rv_BOUNDEDSET-0 integer?)
(define-symbolic overlay_blend_8_rv_BOUNDEDSET-1 integer?)
(define-symbolic overlay_blend_8_rv_BOUNDEDSET-2 integer?)
(define-symbolic overlay_blend_8_rv_BOUNDEDSET-3 integer?)
(define overlay_blend_8_rv (take (list (list overlay_blend_8_rv_BOUNDEDSET-0 overlay_blend_8_rv_BOUNDEDSET-1) (list overlay_blend_8_rv_BOUNDEDSET-2 overlay_blend_8_rv_BOUNDEDSET-3)) overlay_blend_8_rv_BOUNDEDSET-len))
(define-symbolic pixel integer?)
(define-symbolic row integer?)
(define-symbolic row_vec_BOUNDEDSET-len integer?)
(define-symbolic row_vec_BOUNDEDSET-0 integer?)
(define-symbolic row_vec_BOUNDEDSET-1 integer?)
(define row_vec (take (list row_vec_BOUNDEDSET-0 row_vec_BOUNDEDSET-1) row_vec_BOUNDEDSET-len))
(current-bitwidth 6)
(define (assertions)
 (assert (&& (&& (&& (&& (=> (&& (&& (> (matrix-length base ) 1 ) (equal? (matrix-length base ) (matrix-length active ) ) ) (equal? (length (matrix-ref-noerr base 0 ) ) (length (matrix-ref-noerr active 0 ) ) ) ) (overlay_blend_8_inv0 active (matrix-empty ) base 0 0 0 (list-empty )) ) (=> (&& (&& (&& (&& (< row (matrix-length base ) ) (> (matrix-length base ) 1 ) ) (equal? (matrix-length base ) (matrix-length active ) ) ) (equal? (length (matrix-ref-noerr base 0 ) ) (length (matrix-ref-noerr active 0 ) ) ) ) (overlay_blend_8_inv0 active agg.result base col pixel row row_vec) ) (overlay_blend_8_inv1 active base 0 pixel (list-empty ) agg.result row) ) ) (=> (or (&& (&& (&& (&& (&& (&& (&& (>= (list-ref-noerr (matrix-ref-noerr base row ) col ) 16 ) (< col (length (matrix-ref-noerr base 0 ) ) ) ) (< row (matrix-length base ) ) ) (> (matrix-length base ) 1 ) ) (equal? (matrix-length base ) (matrix-length active ) ) ) (equal? (length (matrix-ref-noerr base 0 ) ) (length (matrix-ref-noerr active 0 ) ) ) ) (overlay_blend_8_inv0 active agg.result base col pixel row row_vec) ) (overlay_blend_8_inv1 active base col pixel row_vec agg.result row) ) (&& (&& (&& (&& (&& (&& (&& (! (>= (list-ref-noerr (matrix-ref-noerr base row ) col ) 16 ) ) (< col (length (matrix-ref-noerr base 0 ) ) ) ) (< row (matrix-length base ) ) ) (> (matrix-length base ) 1 ) ) (equal? (matrix-length base ) (matrix-length active ) ) ) (equal? (length (matrix-ref-noerr base 0 ) ) (length (matrix-ref-noerr active 0 ) ) ) ) (overlay_blend_8_inv0 active agg.result base col pixel row row_vec) ) (overlay_blend_8_inv1 active base col pixel row_vec agg.result row) ) ) (overlay_blend_8_inv1 active base (+ col 1 ) (if (&& (&& (&& (&& (&& (&& (&& (! (>= (list-ref-noerr (matrix-ref-noerr base row ) col ) 16 ) ) (< col (length (matrix-ref-noerr base 0 ) ) ) ) (< row (matrix-length base ) ) ) (> (matrix-length base ) 1 ) ) (equal? (matrix-length base ) (matrix-length active ) ) ) (equal? (length (matrix-ref-noerr base 0 ) ) (length (matrix-ref-noerr active 0 ) ) ) ) (overlay_blend_8_inv0 active agg.result base col pixel row row_vec) ) (overlay_blend_8_inv1 active base col pixel row_vec agg.result row) ) (quotient-noerr (* (* 2 (list-ref-noerr (matrix-ref-noerr base row ) col ) ) (list-ref-noerr (matrix-ref-noerr base row ) col ) ) 32 ) (- (- (+ (* 2 (list-ref-noerr (matrix-ref-noerr base row ) col ) ) (list-ref-noerr (matrix-ref-noerr base row ) col ) ) (quotient-noerr (* (* 2 (list-ref-noerr (matrix-ref-noerr base row ) col ) ) (list-ref-noerr (matrix-ref-noerr base row ) col ) ) 32 ) ) 32 ) ) (list-append row_vec (if (&& (&& (&& (&& (&& (&& (&& (! (>= (list-ref-noerr (matrix-ref-noerr base row ) col ) 16 ) ) (< col (length (matrix-ref-noerr base 0 ) ) ) ) (< row (matrix-length base ) ) ) (> (matrix-length base ) 1 ) ) (equal? (matrix-length base ) (matrix-length active ) ) ) (equal? (length (matrix-ref-noerr base 0 ) ) (length (matrix-ref-noerr active 0 ) ) ) ) (overlay_blend_8_inv0 active agg.result base col pixel row row_vec) ) (overlay_blend_8_inv1 active base col pixel row_vec agg.result row) ) (quotient-noerr (* (* 2 (list-ref-noerr (matrix-ref-noerr base row ) col ) ) (list-ref-noerr (matrix-ref-noerr base row ) col ) ) 32 ) (- (- (+ (* 2 (list-ref-noerr (matrix-ref-noerr base row ) col ) ) (list-ref-noerr (matrix-ref-noerr base row ) col ) ) (quotient-noerr (* (* 2 (list-ref-noerr (matrix-ref-noerr base row ) col ) ) (list-ref-noerr (matrix-ref-noerr base row ) col ) ) 32 ) ) 32 ) ) ) agg.result row) ) ) (=> (&& (&& (&& (&& (&& (&& (! (< col (length (matrix-ref-noerr base 0 ) ) ) ) (< row (matrix-length base ) ) ) (> (matrix-length base ) 1 ) ) (equal? (matrix-length base ) (matrix-length active ) ) ) (equal? (length (matrix-ref-noerr base 0 ) ) (length (matrix-ref-noerr active 0 ) ) ) ) (overlay_blend_8_inv0 active agg.result base col pixel row row_vec) ) (overlay_blend_8_inv1 active base col pixel row_vec agg.result row) ) (overlay_blend_8_inv0 active (matrix-append agg.result row_vec ) base col pixel (+ row 1 ) row_vec) ) ) (=> (or (&& (&& (&& (&& (! (< row (matrix-length base ) ) ) (> (matrix-length base ) 1 ) ) (equal? (matrix-length base ) (matrix-length active ) ) ) (equal? (length (matrix-ref-noerr base 0 ) ) (length (matrix-ref-noerr active 0 ) ) ) ) (overlay_blend_8_inv0 active agg.result base col pixel row row_vec) ) (&& (&& (&& (&& (&& (! true ) (! (< row (matrix-length base ) ) ) ) (> (matrix-length base ) 1 ) ) (equal? (matrix-length base ) (matrix-length active ) ) ) (equal? (length (matrix-ref-noerr base 0 ) ) (length (matrix-ref-noerr active 0 ) ) ) ) (overlay_blend_8_inv0 active agg.result base col pixel row row_vec) ) ) (overlay_blend_8_ps base active agg.result) ) )))


    (define sol0
        (synthesize
            #:forall (list active_BOUNDEDSET-len active_BOUNDEDSET-0 active_BOUNDEDSET-1 active_BOUNDEDSET-2 active_BOUNDEDSET-3 agg.result_BOUNDEDSET-len agg.result_BOUNDEDSET-0 agg.result_BOUNDEDSET-1 agg.result_BOUNDEDSET-2 agg.result_BOUNDEDSET-3 base_BOUNDEDSET-len base_BOUNDEDSET-0 base_BOUNDEDSET-1 base_BOUNDEDSET-2 base_BOUNDEDSET-3 col int_x int_y overlay_blend_8_rv_BOUNDEDSET-len overlay_blend_8_rv_BOUNDEDSET-0 overlay_blend_8_rv_BOUNDEDSET-1 overlay_blend_8_rv_BOUNDEDSET-2 overlay_blend_8_rv_BOUNDEDSET-3 pixel row row_vec_BOUNDEDSET-len row_vec_BOUNDEDSET-0 row_vec_BOUNDEDSET-1)
            #:guarantee (assertions)
        )
    )
    (sat? sol0)
    (print-forms sol0)
