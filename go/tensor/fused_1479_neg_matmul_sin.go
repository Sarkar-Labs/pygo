package tensor

import "fmt"

func Fused1479NegMatmulSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_matmul_sin: nil operand")
	}
	r := x.MatMul(y)
	r = r.Neg()
	r = r.Sin()
	return r, nil
}
func Fused1479NegMatmulSinName() string { return "neg_matmul_sin" }
