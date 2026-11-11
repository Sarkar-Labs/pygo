package tensor

import "fmt"

func Fused1459NegMatmulAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_matmul_abs: nil operand")
	}
	r := x.MatMul(y)
	r = r.Neg()
	r = r.Abs()
	return r, nil
}
func Fused1459NegMatmulAbsName() string { return "neg_matmul_abs" }
