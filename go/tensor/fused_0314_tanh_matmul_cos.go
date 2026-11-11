package tensor

import "fmt"

func Fused0314TanhMatmulCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_matmul_cos: nil operand")
	}
	r := x.MatMul(y)
	r = r.Tanh()
	r = r.Cos()
	return r, nil
}
func Fused0314TanhMatmulCosName() string { return "tanh_matmul_cos" }
