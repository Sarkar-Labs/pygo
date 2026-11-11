package tensor

import "fmt"

func Fused0209SigmoidMatmulExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_matmul_exp: nil operand")
	}
	r := x.MatMul(y)
	r = r.Sigmoid()
	r = r.Exp()
	return r, nil
}
func Fused0209SigmoidMatmulExpName() string { return "sigmoid_matmul_exp" }
