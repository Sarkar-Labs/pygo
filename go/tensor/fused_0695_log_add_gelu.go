package tensor

import "fmt"

func Fused0695LogAddGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_add_gelu: nil operand")
	}
	r := x.Add(y)
	r = r.Log()
	r = r.GELU()
	return r, nil
}
func Fused0695LogAddGeluName() string { return "log_add_gelu" }
