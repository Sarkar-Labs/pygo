package tensor

import "fmt"

func Fused0299TanhMatmulExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_matmul_exp: nil operand")
	}
	r := x.MatMul(y)
	r = r.Tanh()
	r = r.Exp()
	return r, nil
}
func Fused0299TanhMatmulExpName() string { return "tanh_matmul_exp" }
