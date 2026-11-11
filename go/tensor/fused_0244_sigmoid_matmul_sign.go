package tensor

import "fmt"

func Fused0244SigmoidMatmulSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_matmul_sign: nil operand")
	}
	r := x.MatMul(y)
	r = r.Sigmoid()
	r = r.Sign()
	return r, nil
}
func Fused0244SigmoidMatmulSignName() string { return "sigmoid_matmul_sign" }
