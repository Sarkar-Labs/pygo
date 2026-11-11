package tensor

import "fmt"

func Fused0119ReluMatmulExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_matmul_exp: nil operand")
	}
	r := x.MatMul(y)
	r = r.ReLU()
	r = r.Exp()
	return r, nil
}
func Fused0119ReluMatmulExpName() string { return "relu_matmul_exp" }
