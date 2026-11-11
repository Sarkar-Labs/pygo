package tensor

import "fmt"

func Fused0339TanhMatmulGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_matmul_gelu: nil operand")
	}
	r := x.MatMul(y)
	r = r.Tanh()
	r = r.GELU()
	return r, nil
}
func Fused0339TanhMatmulGeluName() string { return "tanh_matmul_gelu" }
