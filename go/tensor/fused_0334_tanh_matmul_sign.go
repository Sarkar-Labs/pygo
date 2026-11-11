package tensor

import "fmt"

func Fused0334TanhMatmulSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_matmul_sign: nil operand")
	}
	r := x.MatMul(y)
	r = r.Tanh()
	r = r.Sign()
	return r, nil
}
func Fused0334TanhMatmulSignName() string { return "tanh_matmul_sign" }
