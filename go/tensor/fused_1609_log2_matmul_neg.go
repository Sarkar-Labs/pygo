package tensor

import "fmt"

func Fused1609Log2MatmulNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_matmul_neg: nil operand")
	}
	r := x.MatMul(y)
	r = r.Log2()
	r = r.Neg()
	return r, nil
}
func Fused1609Log2MatmulNegName() string { return "log2_matmul_neg" }
