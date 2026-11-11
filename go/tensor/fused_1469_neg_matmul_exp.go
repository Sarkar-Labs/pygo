package tensor

import "fmt"

func Fused1469NegMatmulExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_matmul_exp: nil operand")
	}
	r := x.MatMul(y)
	r = r.Neg()
	r = r.Exp()
	return r, nil
}
func Fused1469NegMatmulExpName() string { return "neg_matmul_exp" }
