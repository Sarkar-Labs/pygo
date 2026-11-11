package tensor

import "fmt"

func Fused1464NegMatmulSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_matmul_sqrt: nil operand")
	}
	r := x.MatMul(y)
	r = r.Neg()
	r = r.Sqrt()
	return r, nil
}
func Fused1464NegMatmulSqrtName() string { return "neg_matmul_sqrt" }
