package tensor

import "fmt"

func Fused1494NegMatmulCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_matmul_ceil: nil operand")
	}
	r := x.MatMul(y)
	r = r.Neg()
	r = r.Ceil()
	return r, nil
}
func Fused1494NegMatmulCeilName() string { return "neg_matmul_ceil" }
