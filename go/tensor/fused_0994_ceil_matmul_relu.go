package tensor

import "fmt"

func Fused0994CeilMatmulRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_matmul_relu: nil operand")
	}
	r := x.MatMul(y)
	r = r.Ceil()
	r = r.ReLU()
	return r, nil
}
func Fused0994CeilMatmulReluName() string { return "ceil_matmul_relu" }
