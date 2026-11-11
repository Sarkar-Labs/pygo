package tensor

import "fmt"

func Fused0154ReluMatmulSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_matmul_sign: nil operand")
	}
	r := x.MatMul(y)
	r = r.ReLU()
	r = r.Sign()
	return r, nil
}
func Fused0154ReluMatmulSignName() string { return "relu_matmul_sign" }
