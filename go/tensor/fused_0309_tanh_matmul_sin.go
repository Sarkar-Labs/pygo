package tensor

import "fmt"

func Fused0309TanhMatmulSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_matmul_sin: nil operand")
	}
	r := x.MatMul(y)
	r = r.Tanh()
	r = r.Sin()
	return r, nil
}
func Fused0309TanhMatmulSinName() string { return "tanh_matmul_sin" }
