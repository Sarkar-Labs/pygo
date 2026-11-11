package tensor

import "fmt"

func Fused0604ExpMatmulSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_matmul_sign: nil operand")
	}
	r := x.MatMul(y)
	r = r.Exp()
	r = r.Sign()
	return r, nil
}
func Fused0604ExpMatmulSignName() string { return "exp_matmul_sign" }
